import argparse
import glob
import logging
import os
import random
import shutil
import json

from typing import Dict, List, Tuple
from argparse import Namespace
from click import pass_context

import numpy as np
import sklearn
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,Subset
from tqdm import tqdm, trange
import setproctitle
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    RobertaForMultipleChoice,
    BertForMultipleChoice,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup
)
from models import (
    RobertaForMultipleChoice_MLM,
    DebertaForMultipleChoice_MLM,
    DebertaForMultipleChoice,
    BertForMultipleChoice_MLM
)

from dataset_FP_SL_ecom import (
    #jang-r
    BaseDataset,
    SPECIAL_TOKENS
)
#jang-r

from utils.model import (
    #jang
    run_batch_selection_train,
    run_batch_selection_eval,
    run_batch_selection_ranking
)
from utils.data import write_selection_preds, write_detection_preds
# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
#     from tensorboardX import SummaryWriter

# Distributed DataParallel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import wandb
import os
#debug
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="10"


os.environ["TOKENIZERS_PARALLELISM"] = "true"


#debug
dataset_path={
    "train":"ecom/data/train_FP_SNDS.pkl",
    # "train":"new_data/train_SNDS.pkl",
    "val":"ecom/data/valid_mdns.pkl",
    "test":"ecom/data/test_mdns.pkl",
}
model_path={
    "bert":"bert-base-uncased",
    "bert_ch":"bert-base-chinese"
}

def cleanup():
    dist.destroy_process_group()

def get_classes(args):
    #jang-r
    dataset_class=BaseDataset
    if args.multi_task:
      
        model_class=BertForMultipleChoice_MLM
    else:
        model_class=BertForMultipleChoice

                
            
    return dataset_class, model_class, run_batch_selection_train, run_batch_selection_eval,run_batch_selection_ranking


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list

def train_valid_sample(args, train_dataset, indices, model, run_batch_fn_train, optimizer, scheduler, desc="train_valid_sample"):
    valid_loss=0
    valid_steps=0
    train_subset=Subset(train_dataset, indices=indices)
    train_subset_sampler = SequentialSampler(train_subset) if args.local_rank == -1 else DistributedSampler(train_subset,shuffle=False,drop_last=False)
    train_subset_dataloader = DataLoader(
        train_subset,
        sampler=train_subset_sampler,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn
    )
    # for step, batch in enumerate(tqdm(train_subset_dataloader, desc="train_valid", disable=args.local_rank not in [-1, 0])):
    for step, batch in enumerate(train_subset_dataloader):
    
        model.train()
        
        loss, _, _ = run_batch_fn_train(args, model, batch)

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
            
        loss.backward()
        valid_loss += loss.item()
        #볼것
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            valid_steps += 1      

    return valid_loss, valid_steps

def train(args, train_dataset, ranking_dataset, val_dataset,test_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, run_batch_fn_train, run_batch_fn_eval,run_batch_fn_ranking) -> Tuple[int, float]:
    if args.local_rank in [-1, 0]:
        args.output_dir = args.exp_name 

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    if args.local_rank != -1:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )

    #jang
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps
    )
    
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    # )

    # Train!
    global_step = 0
    model.zero_grad()
    train_iterator = trange(0, int(args.epochs), desc="Epoch",disable=args.local_rank not in [-1, 0])
    previous_best=0
    data_loader_indics=list(range(len(train_dataset))) # 데이터 길이 만큼의 list 생성
    update_span=args.ranking_step*args.batch_size

    for epoch in train_iterator:
        local_steps = 0
        tr_loss = 0.0
        random.shuffle(data_loader_indics)
        
        if args.local_rank not in [-1]:
            data_loader_indics=all_gather(data_loader_indics)[0]
        
        
        data_start=0
        data_end=update_span

        valid_sample=0
        ranking_tr=0
        ranking_fl=0
        if args.negative_sample_method=='ranking' and (epoch>=1 or args.checkpoint):
            if len(data_loader_indics)%update_span!=0:
                train_with_ranker_len=len(data_loader_indics)//update_span+1
            else:
                train_with_ranker_len=len(data_loader_indics)//update_span
            
            stacked_negative_dict={}
            for step in tqdm(range(train_with_ranker_len), desc='train_with_ranker', disable=args.local_rank not in [-1, 0]):
                #debug
                
                data_end=min(data_end,len(data_loader_indics)+1)
                indices=data_loader_indics[data_start:data_end]

                negative_dict, tr_cnt, fl_cnt=ranker(args, ranking_dataset, indices, model, run_batch_fn_ranking, desc=str(global_step))
                

                if args.local_rank != -1:
                    output=all_gather(negative_dict)
                    tr_cnt=sum(all_gather(tr_cnt))
                    fl_cnt=sum(all_gather(fl_cnt))
                    gathered_dict={}
                    for rank_dict in output:
                        gathered_dict.update(rank_dict)
                    negative_dict=gathered_dict
                
                if args.local_rank not in [-1]:
                    torch.distributed.barrier()
                
                ranking_tr+=tr_cnt
                ranking_fl+=fl_cnt
                # train_dataset.negative=negative_dict
                data_start+=update_span
                data_end+=update_span
                valid_sample+=len(list(negative_dict.keys()))
                
                # if args.local_rank in [-1,0]:
                #     print("ranking vs valid_sample= %d vs %d"%(ranking_tr+ranking_fl,valid_sample))

                stacked_negative_dict.update(negative_dict)
                if args.local_rank not in [-1]:
                    torch.distributed.barrier()
                
                
                length=len(list(stacked_negative_dict.keys()))
                
                if length>=args.batch_size*args.world_size :
                    residual=length%(args.batch_size*args.world_size)
                    if residual!=0:
                        train_stacked_negative_dict=dict(list(stacked_negative_dict.items())[:-residual])
                        train_dataset.negative=train_stacked_negative_dict
                        valid_loss, valid_step=train_valid_sample(args, train_dataset, list(train_stacked_negative_dict.keys()), model, run_batch_fn_train, optimizer, scheduler, desc="train_valid_sample")
                        global_step += valid_step
                        local_steps += valid_step
                        tr_loss +=valid_loss
                        stacked_negative_dict=dict(list(stacked_negative_dict.items())[-residual:])
                    else:
                        train_dataset.negative=stacked_negative_dict
                        valid_loss, valid_step=train_valid_sample(args, train_dataset, list(stacked_negative_dict.keys()), model, run_batch_fn_train, optimizer, scheduler, desc="train_valid_sample")
                        global_step += valid_step
                        local_steps += valid_step
                        tr_loss +=valid_loss
                        stacked_negative_dict={}
                 
                    if args.local_rank in [-1, 0] and global_step%100==0 and local_steps!=0: 
                        wandb.log({"lossperstep":tr_loss / local_steps}, step=global_step)  
                 
        
                elif step==train_with_ranker_len-1 and length>=1:
                    train_dataset.negative=stacked_negative_dict
                    valid_loss, valid_step=train_valid_sample(args, train_dataset, list(stacked_negative_dict.keys()), model, run_batch_fn_train, optimizer, scheduler, desc="train_valid_sample")
                    global_step += valid_step
                    local_steps += valid_step
                    tr_loss +=valid_loss
                    stacked_negative_dict={}
                    if args.local_rank in [-1, 0] and global_step%100==0 and local_steps!=0: 
                        wandb.log({"lossperstep":tr_loss / local_steps}, step=global_step)   
                else:
                    debug=0
            
            if args.local_rank not in [-1]:
                torch.distributed.barrier()
            if args.local_rank in [-1, 0]:
                print("\nvalid_sample: %d(%d)"%(valid_sample,len(data_loader_indics)))
                print("train True vs False= %d vs %d"%(ranking_tr,ranking_fl))  
                print("global step %d"%(global_step))  
                
        else:
            valid_sample=len(data_loader_indics)
            train_subset=Subset(train_dataset, indices=data_loader_indics)
            train_subset_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset,shuffle=False,drop_last=False)
            train_subset_dataloader = DataLoader(
                train_subset,
                sampler=train_subset_sampler,
                batch_size=args.batch_size,
                collate_fn=train_dataset.collate_fn
            )
            for step,batch in enumerate(tqdm(train_subset_dataloader, desc='train_base', disable=args.local_rank not in [-1, 0])):
                model.train()
                loss, _, _ = run_batch_fn_train(args, model, batch)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    
                loss.backward()
                tr_loss += loss.item()
                #볼것
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    local_steps += 1     
                    if args.local_rank in [-1, 0] and local_steps%100==0: 
                        wandb.log({"lossperstep":tr_loss / local_steps}, step=global_step)   

        # if global_step>100:
        if args.local_rank not in [-1]:
            torch.distributed.barrier()
        if args.local_rank in [-1, 0]:
            print("global step %d"%(global_step))  
            # tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)    
            # tb_writer.add_scalar("loss", tr_loss / local_steps, global_step)
            # tb_writer.add_scalar("valid_sample_num",valid_sample, global_step)
            # tb_writer.add_scalar("ranking_fl",ranking_fl, global_step)
            wandb.log({"lr": scheduler.get_lr()[0],"loss":tr_loss / local_steps}, step=global_step)    
            wandb.log({"valid_sample_num":valid_sample,"ranking_fl":ranking_fl}, step=global_step)

        #debug
        if epoch>=0:
            val_results = evaluate(args, val_dataset, model, tokenizer, run_batch_fn_eval, desc="validation")
            if args.local_rank in [-1, 0]:
                # tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
            
                for key, value in val_results.items():
                    # tb_writer.add_scalar("val_{}".format(key), value, global_step)
                    wandb.log({"val_{}".format(key): value},step=global_step)
                
            if args.local_rank not in [-1]:
                torch.distributed.barrier() 

            val_current_score=val_results['r1']
        
            if val_current_score >= previous_best:
                previous_best=val_current_score
                #best 모델 checkpoint 밖
                if args.local_rank in [-1, 0]:

                    os.makedirs(args.output_dir, exist_ok=True)
                    print("Saving best model checkpoint to %s", args.output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)

                    # # Good practice: save your training arguments together with the trained model
                    # torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
                    # with open(os.path.join(args.output_dir, "params.json"), "w") as jsonfile:
                    #     json.dump(args.params, jsonfile, indent=2)

                    # checkpoint_prefix = "checkpoint"
                    # # Save model checkpoint
                    # output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    # os.makedirs(output_dir, exist_ok=True)
                    # logger.info("Saving model checkpoint to %s", output_dir)
                    # model_to_save = (
                    #     model.module if hasattr(model, "module") else model
                    # )  # Take care of distributed/parallel training

                    # model_to_save.save_pretrained(output_dir)
                    # tokenizer.save_pretrained(output_dir)

                    # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    # with open(os.path.join(output_dir, "params.json"), "w") as jsonfile:
                    #     json.dump(args.params, jsonfile, indent=2, default=lambda x: str(x))
                if args.eval_with_test :
                    test_results = evaluate(args, test_dataset, model, tokenizer, run_batch_fn_eval, desc="test")
                    if args.local_rank not in [-1]:
                        torch.distributed.barrier() 
                    if args.local_rank in [-1, 0]:
                        for key, value in test_results.items():
                            # tb_writer.add_scalar("test_{}".format(key), value, global_step)
                            wandb.log({"test_{}".format(key): value},step=global_step)

    
    if args.local_rank in [-1, 0]:
        # tb_writer.close()
        wandb.finish()

    return global_step, tr_loss / local_steps


def evaluate(args, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, run_batch_fn, desc="") -> Dict:
    # if args.local_rank in [-1, 0]:
    # eval_output_dir = args.output_dir
    # os.makedirs(eval_output_dir, exist_ok=True)

    # eval_batch_size for selection must be 1 to handle variable number of candidates
    args.eval_batch_size = 1
    
    eval_sampler=SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset,shuffle=False)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=eval_dataset.collate_fn,
        # num_workers=2
    )

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=desc,disable=args.local_rank not in [-1, 0]):
            logits, labels = run_batch_fn(args, model, batch)
            all_preds.append(logits.squeeze())
            all_labels.append(labels.squeeze(0).tolist())

        all_pred_ids = [logits.argsort(descending=True).tolist() for logits in all_preds]
    
        if args.local_rank != -1:
            all_pred_ids=[pred_id for pred_ids in all_gather(all_pred_ids) for pred_id in pred_ids]
            all_labels=[label for labels in all_gather(all_labels) for label in labels]

    r1_score=0.0
    r2_score=0.0
    r5_score=0.0

    num_sample=len(all_labels)
    average_candidate_num=0
    for gt, pred in zip(all_labels, all_pred_ids):
        #acc_score
        if isinstance(pred,int):
            pred=[pred]
        
        average_candidate_num+=len(pred)
    

        #mrr_score
        for item in pred[:1]:
            if item == 0:
                r1_score+=1
                break

        for item in pred[:2]:
            if item == 0:
                r2_score+=1
                break

        for item in pred[:5]:
            if item == 0:
                r5_score+=1
                break
        

    r1_score/=num_sample
    r2_score/=num_sample
    r5_score/=num_sample
    
    result = {"r1": r1_score,"r2": r2_score,"r5":r5_score}
    if args.local_rank in [-1, 0]:
        print("Avg. # of candidates: %f\n"%(average_candidate_num/len(all_pred_ids)))
        print("%s\tr1: %.4f\tr2: %.4f\tr5: %.4f\t\n"%(desc,r1_score,r2_score,r5_score))


    # if args.output_file:
    #     sorted_pred_ids = [np.argsort(logits.squeeze())[::-1] for logits in all_preds]
    #     write_selection_preds(eval_dataset.dataset_walker, args.output_file, data_infos, sorted_pred_ids, topk=5)
    
    #jang
    # output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    # with open(output_eval_file, "a") as writer:
    #     logger.info("***** %s results %s *****" %(eval_dataset.split_type, desc))
    #     writer.write("***** %s results %s *****\n" %(eval_dataset.split_type, desc))
    #     for key in sorted(result.keys()):
    #         logger.info("  %s = %s", key, str(result[key]))
    #         writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def ranker(args, ranking_dataset, indices, model: PreTrainedModel, run_batch_fn, desc="") -> Dict:
    # if args.local_rank in [-1, 0]:
    #     eval_output_dir = args.output_dir
    #     os.makedirs(eval_output_dir, exist_ok=True)

    args.sample_batch_size = 1

    ranking_subset=Subset(ranking_dataset, indices=indices)
    ranking_subset_sampler = SequentialSampler(ranking_subset) if args.local_rank == -1 else DistributedSampler(ranking_subset,shuffle=False)
    ranking_dataloader = DataLoader(
        ranking_subset,
        sampler=ranking_subset_sampler,
        batch_size=args.sample_batch_size,
        collate_fn=ranking_dataset.collate_fn
    )

    model.eval()
    negative_cand_dict={}
    tr_cnt=0
    fl_cnt=0
    with torch.no_grad():
        # for batch in tqdm(ranking_dataloader, desc="ranking", disable=args.local_rank not in [-1, 0]):
        for batch in ranking_dataloader:
            mc_logits = run_batch_fn(args, model, batch)
            #정답 인덱스 찾기
            pred_idx=torch.argmax(mc_logits)
             
            if pred_idx!=0:
                fl_cnt+=1
            else:
                tr_cnt+=1
          
            
            confident=nn.Softmax(dim=-1)(mc_logits)[0].item()
            #ranking_dataset.examples[batch[-1]['sample_id'][0]]
            #ranking_dataset.all_response[16753]
            if confident<=0.99:
            # if True:
            # if pred_idx!=0:
                cand_ids=torch.argsort(mc_logits[1:],descending=True)
                cand_ids=cand_ids[:args.candidates_num]
                cand_ids=np.array(batch[-1]['negative_id'][0])[cand_ids.cpu().numpy()]
                negative_cand_dict[batch[-1]['sample_id'][0]]=cand_ids
            else:
                a=1
            
            # msmarco
            # confidence=nn.Softmax(dim=-1)(mc_logits)[0].item()
            # negative_score=mc_logits[1:]
            # _ , sorted_indices = torch.sort(negative_score, descending=True)
            # # efficient
            # if confidence <= 0.99:
            #     cand_ids=sorted_indices
            #     cand_ids=np.array(batch[-1]['negative_id'][0])[cand_ids.cpu().numpy()]
            #     negative_cand_dict[batch[-1]['sample_id'][0]]=cand_ids


    return  negative_cand_dict,tr_cnt,fl_cnt

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--premodel", type=str, default="bert", 
                        help="pretrained model bert,roberta,albert,deberta")
    parser.add_argument("--eval_only", action="store_true",default=False,
                        help="Perform evaluation only")
    parser.add_argument("--checkpoint", type=str, help="Saved checkpoint directory")
    parser.add_argument("--max_seq", type=int, default=512,
                        help="Maximum length in tokens for history, will override that value in config.")
    parser.add_argument("--output_file", type=str, default="", help="Predictions will be written to this file.")
    parser.add_argument("--negative_sample_method", type=str, choices=["random", "ranking"],
                        default="ranking", help="Negative sampling method for knowledge selection, will override the value in config.")
    parser.add_argument("--eval_with_test", default=False, action='store_true',
                        help="If set, the candidates to be selected would be all knowledge snippets, not sampled subset.") 
    parser.add_argument("--exp_name", type=str, default="test",
                        help="Name of the experiment, checkpoints will be stored in runs/{exp_name}")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--negative_num", type=int, default=3,
                        help="negative cand number")
    parser.add_argument("--candidates_num", type=int, default=20,
                        help="select candidate number and rerank")
    parser.add_argument("--batch_size", type=int, default=4 ,help="batch_size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,help="gradient_accumulation")

    parser.add_argument("--candidate_batch_size", type=int, default=256,help="batch_size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--multi_task", default=False, action='store_true', help="If set, multitask true.")
    parser.add_argument("--epochs", type=int, default=10,help="epochs")
    
    parser.add_argument("--warmup_steps", type=int, default=0,help="warmup")
    parser.add_argument("--max_grad_norm", type=int, default=10,
                        help="grad clip")
    
    #51993
    parser.add_argument("--ranking_step", type=int, default=16,help="rank update")

    # Distriution 
    parser.add_argument('--local_rank', type=int, default=-1, help='Local process rank.')
    parser.add_argument("--train_path", type=str, default="new_data/train_FP_SNDS_100k.pkl", 
                        help="trainset_path")
    parser.add_argument("--val_path", type=str, default="new_data/train_FP_SNDS_100k.pkl", 
                        help="trainset_path")
    parser.add_argument("--test_path", type=str, default="new_data/train_FP_SNDS_100k.pkl", 
                        help="trainset_path")
    parser.add_argument("--res_path", type=str, default="new_data/all_response.pkl", 
                            help="negative_response_path")
    # parser.add_argument('--rank', type=int, default=-1, help='rank') 
    # parser.add_argument('--world_size', type=int, default=-1, help='world size') 

  


    args = parser.parse_args()
    setproctitle.setproctitle("janghoon_"+args.exp_name)
    if args.local_rank in [-1, 0]:
        logger = logging.getLogger(__name__)
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO
        )
        print(args)

    if args.local_rank in [-1, 0]:
        wandb.init(
        # set the wandb project where this run will be logged
        project=args.exp_name.split('/')[-1],
        # name="test",
        # track hyperparameters and run metadata
        config=args
        )


    args.distributed = (args.local_rank != -1)
    if not args.distributed:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        args.world_size=dist.get_world_size()
    #jang 
    # args.n_gpu=1
    # args.device = device
    args.model_name_or_path=model_path[args.premodel]
    # print("gpu num %d"%args.n_gpu)
    dataset_path['train']=args.train_path
    dataset_path['val']=args.val_path
    dataset_path['test']=args.test_path
    # Set seed
    set_seed(args)
    

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  

    dataset_class, model_class, run_batch_fn_train, run_batch_fn_eval, run_batch_fn_ranking = get_classes(args)

   

    if args.checkpoint:
        args.output_dir = args.checkpoint
        model = model_class.from_pretrained(args.checkpoint)
        if args.local_rank in [-1, 0]:
            print(model_class)
            print("%s : load"%(args.checkpoint))
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
       
        tokenizer.add_special_tokens({'eos_token': '[eos]'})

    else:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        # set output_past to False for DataParallel to work during evaluation
        config.output_past = False
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        if args.local_rank in [-1, 0]:
            print(len(tokenizer))

        tokenizer.add_special_tokens({'eos_token': '[eos]'})

        model = model_class.from_pretrained(args.model_name_or_path, config=config)
        model.resize_token_embeddings(len(tokenizer))
        
        #bertfp 여기서 켯다 끄기
        # if args.premodel=="bert":
        #     map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
        #     model.bert.load_state_dict(state_dict=torch.load("bertfp_ckpt/bert.pt",map_location=map_location),strict=False)

        # if args.local_rank in [-1, 0]:
        #     print("%s : load"%(args.model_name_or_path))
        #     if args.premodel=="bert":
        #         print("bertfp_ckpt/bert.pt : load")



        if args.premodel=="bert_ch":
            map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
            model.bert.load_state_dict(state_dict=torch.load("e-com/pt_ckpt/bert.pt",map_location=map_location),strict=False)

        if args.local_rank in [-1, 0]:
            print("%s : load"%(args.model_name_or_path))
            if args.premodel=="bert_ch":
                print("e-com/pt_ckpt/bert.pt : load")

        
    model = model.to(device)
    
    if args.local_rank == 0:
        torch.distributed.barrier() 
    
    if args.local_rank in [-1, 0]:
        logger.info("Training/evaluation parameters %s", args)
    
    if not args.eval_only:
        #jang
        train_dataset = dataset_class(dataset_path['train'], args, tokenizer, type="train")
        
        if args.negative_sample_method=='ranking':
            ranking_dataset = dataset_class(dataset_path['train'], args, tokenizer, type="ranking")
        else:
            ranking_dataset = None
        
        val_dataset = dataset_class(dataset_path['val'],args, tokenizer, type="val")
        
        if args.eval_with_test:
            test_dataset = dataset_class(dataset_path['test'], args, tokenizer, type="test")
        else:
            test_dataset=None
        
        

        global_step, tr_loss = train(args, train_dataset, ranking_dataset, val_dataset, test_dataset, model, tokenizer, run_batch_fn_train, run_batch_fn_eval,run_batch_fn_ranking)
        if args.local_rank in [-1, 0]:
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        #jang
        
        # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
       
        '''
        os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        with open(os.path.join(args.output_dir, "params.json"), "w") as jsonfile:
            json.dump(params, jsonfile, indent=2)
        '''
        # Load a trained model and vocabulary that you have fine-tuned
        
        # model = model_class.from_pretrained(args.output_dir)
        # tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        # model.to(args.device)
        
    # Evaluation
    result = {}
    
    if args.eval_only:
        test_dataset = dataset_class(dataset_path['test'], args, tokenizer, type="test")
        test_results = evaluate(args, test_dataset, model, tokenizer, run_batch_fn_eval, desc="test")
        if args.local_rank not in [-1]:
            torch.distributed.barrier() 

    return result


if __name__ == "__main__":
    main()
