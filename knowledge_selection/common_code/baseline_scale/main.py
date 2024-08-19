import argparse
import glob
import logging
import os
import random
from typing import Dict, List, Tuple
from argparse import Namespace

import numpy as np
import torch

import setproctitle

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    RobertaForMultipleChoice,
    BertForMultipleChoice,
    AlbertForMultipleChoice,
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

from dataset import (
    KnowledgeSelectionDataset,
    SPECIAL_TOKENS
)
#jang-r

from utils.model import (
    #jang
    run_batch_selection_train,
    run_batch_selection_eval,
    run_batch_selection_ranking
)
from utils.data import write_selection_preds


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# Distributed DataParallel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)

import os
#debug
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"]="12"


os.environ["TOKENIZERS_PARALLELISM"] = "true"



def cleanup():
    dist.destroy_process_group()


def get_classes(args):
  
    #jang-r
    if args.task.lower() == "selection":
        dataset_class=KnowledgeSelectionDataset
        if args.multi_task:
            if args.premodel=='roberta':
                model_class=RobertaForMultipleChoice_MLM
            elif args.premodel=='deberta':
                model_class=DebertaForMultipleChoice_MLM
        else:
            if args.premodel=='roberta':
                    model_class=RobertaForMultipleChoice
            elif args.premodel=='deberta':
                model_class=DebertaForMultipleChoice   
                
            
        return dataset_class, model_class , run_batch_selection_train, run_batch_selection_eval,run_batch_selection_ranking
    

    else:
        raise ValueError("args.task not in ['selection'], got %s" % args.task)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def all_gather(data):
    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def train(args, train_dataset, ranking_dataset, val_dataset,test_seen_dataset,test_unseen_entity_dataset,test_unseen_domain_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, run_batch_fn_train, run_batch_fn_eval,run_batch_fn_ranking) -> Tuple[int, float]:
    if args.local_rank in [-1, 0]:
        log_dir = os.path.join("runs", args.exp_name) if args.exp_name else None
        tb_writer = SummaryWriter(log_dir)
        args.output_dir = log_dir

    args.train_batch_size = args.batch_size

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = DDP(
            model, 
            device_ids=[args.local_rank], 
            output_device=args.local_rank, 
            # find_unused_parameters=True
        )

    # Train!
    global_step = 0
    model.zero_grad()
    train_iterator = trange(
        0, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    previous_best=0
    #debug
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps
    )
    # if args.local_rank==-1:
    #     t_total=(len(data_loader_indics)*args.num_train_epochs//args.batch_size)+1
    # else:
    #     t_total=(len(data_loader_indics)*args.num_train_epochs//(dist.get_world_size()*args.batch_size))+1
    
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    # ) 

    for epoch in train_iterator:
        local_steps = 0
        tr_loss = 0.0
     
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset,shuffle=True,drop_last=False)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            collate_fn=train_dataset.collate_fn
        )
        for step,batch in enumerate(tqdm(train_dataloader, desc='train_base', disable=args.local_rank not in [-1, 0])):
            model.train()
            loss, _, _ = run_batch_fn_train(args, model, batch)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                
            loss.backward()
            tr_loss += loss.item()
            #볼것
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                local_steps += 1      

        # if global_step>100:
        if args.local_rank not in [-1]:
            torch.distributed.barrier()
        if args.local_rank in [-1, 0]:
            tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)    
            tb_writer.add_scalar("loss", tr_loss / local_steps, global_step)
        #debug
        if epoch>=0:
            output_dir=""
            if args.local_rank in [-1, 0]:
                checkpoint_prefix = "checkpoint"
                output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                os.makedirs(output_dir, exist_ok=True)
                logger.info("Saving checkpoint to %s", output_dir)

            if args.local_rank in [-1, 0]:
                # Good practice: save your training arguments together with the trained model
                
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

            val_results = evaluate(args, val_dataset, model, run_batch_fn_eval, output_dir, desc="val")

            if args.local_rank not in [-1]:
                torch.distributed.barrier()
            if args.local_rank in [-1, 0]:
                for key, value in val_results.items():
                    tb_writer.add_scalar("val_{}".format(key), value, global_step)

            
            val_current_score=val_results['accuracy']
        
            if val_current_score > previous_best:
                previous_best=val_current_score
               
                if args.eval_with_test:
                    test_results1 = evaluate(args, test_seen_dataset, model, run_batch_fn_eval, output_dir, desc="test_part")
                    if args.local_rank not in [-1]:
                        torch.distributed.barrier() 
                    if args.local_rank in [-1, 0]:
                        tb_writer.add_scalar("test_part_{}".format('accuracy'), test_results1['accuracy'], global_step)
                    
    if args.local_rank not in [-1]:
        torch.distributed.barrier()
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / local_steps


def evaluate(args, eval_dataset, model: PreTrainedModel, run_batch_fn, output_dir, desc="") -> Dict:
    if args.local_rank in [-1, 0]:
        eval_output_dir = output_dir

    # eval_batch_size for selection must be 1 to handle variable number of candidates
    if args.task == "selection":
        args.eval_batch_size = 1
    else:
        args.eval_batch_size = args.max_candidates_per_forward_eval 

    eval_sampler=SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset,shuffle=False)
    
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=eval_dataset.collate_fn,
        # num_workers=2
    )

    
    model.eval()
    data_infos = []
    all_preds = []
    all_labels = []
    for batch in tqdm(eval_dataloader, desc=desc, disable=args.local_rank not in [-1, 0]):
        with torch.no_grad():
            loss, mc_logits, mc_labels = run_batch_fn(args, model, batch)
            if args.task in ["selection"]:
                data_infos.append(batch[-1])
            all_preds.append(mc_logits.detach().cpu().numpy())
            all_labels.append(mc_labels.detach().cpu().numpy())


    if args.local_rank != -1:
        all_preds=[pred for preds in all_gather(all_preds) for pred in preds]
        all_labels=[label for labels in all_gather(all_labels) for label in labels]
        data_infos=[data_info for each_data_infos in all_gather(data_infos) for data_info in each_data_infos]
        

    if args.task.lower() == "selection":
        
        all_labels = np.array(all_labels).reshape(-1)
        all_pred_ids = np.array([np.argmax(logits) for logits in all_preds])
        accuracy = np.sum(all_pred_ids == all_labels) / len(all_labels)

        logger.info("Avg. # of candidates: %f", sum([len(arr[0]) for arr in all_preds]) / len(all_preds))
        result = {"accuracy": accuracy,"length":len(all_labels)}
        if args.local_rank in [-1, 0]:  
            output_file=eval_output_dir+'/pred_'+desc+'.json'
            sorted_pred_ids = [np.argsort(logits.squeeze())[::-1] for logits in all_preds]
            write_selection_preds(eval_dataset.dataset_walker, output_file, data_infos, sorted_pred_ids, topk=5)
    
   
    else:
        raise ValueError("args.task not in ['selection'], got %s" % args.task)

    if args.local_rank in [-1, 0]:    
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** %s results *****" %(output_dir))
            writer.write("***** %s results *****\n" %(output_dir))
            for key in sorted(result.keys()):
                logger.info("%s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return result



def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    
    parser.add_argument("--dataroot", type=str, default="data",
                        help="Path to dataset.")
    parser.add_argument("--history_max_tokens", type=int, default=384,
                        help="Maximum length in tokens for history, will override that value in config.")
    parser.add_argument("--knowledge_max_tokens", type=int, default=128,
                        help="Maximum length in tokens for knowledge, will override that value in config.")
    parser.add_argument("--knowledge_file", type=str, default="knowledge.json",
                        help="knowledge file name.")
    parser.add_argument("--no_labels", action="store_true",
                        help="Read a dataset without labels.json. This option is useful when running "
                             "knowledge-seeking turn detection on test dataset where labels.json is not available.")
    parser.add_argument("--labels_file", type=str, default=None,
                        help="If set, the labels will be loaded not from the default path, but from this file instead."
                        "This option is useful to take the outputs from the previous task in the pipe-lined evaluation.")

    parser.add_argument("--eval_only", action="store_true",help="Perform evaluation only")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--max_grad_norm", type=float, default=5.0 ,help="batch_size")                    


    parser.add_argument("--negative_sample_method", type=str, choices=["random", "scale","bm25"],
                        default="scale", help="Negative sampling method for knowledge selection, will override the value in config.")
    parser.add_argument("--selection_type", type=str, choices=["all", "body"],
                        default="all", help="Negative sampling method for knowledge selection, will override the value in config.")
    parser.add_argument("--eval_with_test", default=False, action='store_true',
                        help="If set, the candidates to be selected would be all knowledge snippets, not sampled subset.") 
    parser.add_argument("--exp_name", type=str, default="test",
                        help="Name of the experiment, checkpoints will be stored in runs/{exp_name}")
    
    parser.add_argument("--task", type=str, default="selection",choices=["selection"], help="pretrained model bert,roberta,albert,deberta")
    parser.add_argument("--checkpoint", type=str, help="Saved checkpoint directory")
    parser.add_argument("--premodel", type=str, default="roberta", help="pretrained model bert,roberta,albert,deberta")
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="checkpoint for post-training")
    parser.add_argument("--warmup_steps", type=int, default=0 ,help="batch_size")
    

    
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
   
    
    parser.add_argument("--negative_num", type=int, default=1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--candidate_num", type=int, default=64,
                        help="Local rank for distributed training (-1: not distributed)")
    
    
    parser.add_argument("--batch_size", type=int, default=4 ,help="batch_size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8 ,help="batch_size")
    parser.add_argument("--max_candidates_per_forward_eval", type=int, default=256 ,help="batch_size")
    
    parser.add_argument("--num_train_epochs", type=int, default=50 ,help="batch_size")
    parser.add_argument("--seed", type=int, default=0 ,help="batch_size")

    parser.add_argument("--multi_task", action='store_true', help="If set, multitask true.")


    args = parser.parse_args()

    
    
    setproctitle.setproctitle(args.exp_name)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    # Setup CUDA, GPU & distributed training
    args.distributed = (args.local_rank != -1)
    if not args.distributed:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    
    
    #jang 
    args.n_gpu=1
    args.device = device

    
    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab
    
    if args.task.lower()=='selection':
        dataset_class, model_class, run_batch_fn_train, run_batch_fn_eval,run_batch_fn_ranking = get_classes(args)
    
        
       

    if args.checkpoint:
        args.output_dir = args.checkpoint
        model = model_class.from_pretrained(args.checkpoint)
        if args.local_rank in [-1, 0]:
            print(model_class)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        if args.task.lower()=='selection':
            # init_special_tokens_by_model(tokenizer)  
            tokenizer.add_special_tokens(SPECIAL_TOKENS)
            model.resize_token_embeddings(len(tokenizer))
        if args.local_rank in [-1, 0]:
            print("%s : load"%(args.checkpoint))
    else:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        # set output_past to False for DataParallel to work during evaluation
        config.output_past = False
 
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        #jang-r
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
        model.resize_token_embeddings(len(tokenizer))
        if args.local_rank in [-1, 0]:
            print("%s : load"%(args.model_name_or_path))

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    args.vocab_size = model.config.vocab_size
    args.type_vocab_size = model.config.type_vocab_size
  
    if args.local_rank in [-1, 0]:
        logger.info("Training/evaluation parameters %s", args)
    if not args.eval_only:
        #jang
        train_dataset = dataset_class(args, tokenizer, split_type="train")

        if args.negative_sample_method=='ranking' and args.task.lower()=='selection':
            ranking_dataset = dataset_class(args, tokenizer, split_type="ranking")
        else:
            ranking_dataset = None
        
        val_dataset = dataset_class(args, tokenizer, split_type="val")
        
        #jang test knowledge 바뀌니까 마지막
        if args.eval_with_test:
            #debug
            if "test" not in args.knowledge_file:
                if os.path.exists(os.path.join(args.dataroot,"test/"+args.knowledge_file)):
                    args.knowledge_file = "test/"+args.knowledge_file
            
            test_seen_dataset = dataset_class(args, tokenizer, split_type="test")
            test_unseen_entity_dataset = None
            test_unseen_domain_dataset = None
            # test_unseen_entity_dataset = dataset_class(args, tokenizer, split_type="test_unseen_entity")
            # test_unseen_domain_dataset = dataset_class(args, tokenizer, split_type="test_unseen_domain")
        
        else:
            test_seen_dataset = None
            test_unseen_entity_dataset = None
            test_unseen_domain_dataset = None

        
        if args.task.lower()=='selection':
            global_step, tr_loss = train(args, train_dataset, ranking_dataset, val_dataset, test_seen_dataset,
                                    test_unseen_entity_dataset,test_unseen_domain_dataset, \
                                    model, tokenizer, run_batch_fn_train, run_batch_fn_eval,run_batch_fn_ranking)
    
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        #jang
        
        # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
        if args.local_rank in [-1, 0]:
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
    else:
        #debug
        # args.output_dir="test"
        #test
        output_dir=args.checkpoint
        # val_dataset = dataset_class(args, tokenizer, split_type="val")
        # evaluate(args, val_dataset, model, run_batch_fn_eval, output_dir, desc="val")
        
        if "test" not in args.knowledge_file:
            if os.path.exists(os.path.join(args.dataroot,"test/"+args.knowledge_file)):
                    args.knowledge_file = "test/"+args.knowledge_file
        # eval_dataset = dataset_class(args, tokenizer, split_type="test", labels=not args.no_labels, labels_file=args.labels_file)
        # evaluate(args, eval_dataset, model, tokenizer, run_batch_fn_eval, desc=args.eval_desc or "test")
        
        
        # test_dataset = dataset_class(args, tokenizer, split_type="test")
        # test_unseen_entity_dataset = dataset_class(args, tokenizer, split_type="test_unseen_entity")
        # evaluate(args, test_unseen_entity_dataset, model, run_batch_fn_eval, output_dir, desc="test_unseen_entity")
        # test_unseen_domain_dataset = dataset_class(args, tokenizer, split_type="test_unseen_domain")
        # evaluate(args, test_unseen_domain_dataset, model, run_batch_fn_eval, output_dir, desc="test_unseen_domain")
        test_seen= dataset_class(args, tokenizer, split_type="test_seen")
        evaluate(args, test_unseen_entity_dataset, model, run_batch_fn_eval, output_dir, desc="test_seen")



if __name__ == "__main__":
    main()

