
import random
import torch
import pickle
from utils.data import (
    pad_ids
)
import numpy as np
# np.random.seed(0)
# random.seed(0)
from datasets import load_dataset
SPECIAL_TOKENS = {
    
    "pad_token": "<pad>",
    "additional_special_tokens": ["<sep>","<cls>","[eos]"],
}

class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, data_file, args, tokenizer, type="train" ):
        self.args=args
        self.type=type
        self.tokenizer=tokenizer
        self.SPECIAL_TOKENS = SPECIAL_TOKENS

        self.pad = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])
        self.mask = self.tokenizer.mask_token_id
        self.unk=self.tokenizer.unk_token_id
        self.sep_id = self.tokenizer.convert_tokens_to_ids( self.SPECIAL_TOKENS["additional_special_tokens"])[0]
        self.cls_id = self.tokenizer.convert_tokens_to_ids( self.SPECIAL_TOKENS["additional_special_tokens"])[1]
        
        self.negative=None

       
        # with open(data_file, 'rb') as f:
        #     self.examples=pickle.load(f)

       
        
        if type=='train' or type=='ranking':
            self.examples=load_dataset("json", data_files=data_file, cache_dir=data_file.split(".jsonl")[0])['train']
            with open(args.res_path, 'rb') as f:
                self.all_response=pickle.load(f)

        else:
            with open(data_file, 'rb') as f:
                self.examples=pickle.load(f)
        
        
        # # # debug
        # if type=='train' or type=='ranking':
        #     self.examples=self.examples[:50000]
            
            # with open("data/toy_train.pkl", 'wb') as f:
            #     pickle.dump(self.examples, f)
        # else:
        #     self.examples=self.examples[:10000]


        if args.local_rank in [-1, 0]:    
            print("data type %s and length: %d"%(type,len(self.examples)))

  

    def __getitem__(self, index):
        example = self.examples[index]
        if self.type== 'train' or  self.type=='ranking':
            context=example['context']
            postive_resp=example['pos']
            neg_cand_idx=example['neg_list']
        else:
            context=example[0]
            postive_resp=example[1]
            negative_resp_list=example[-1]
       
        sample_id=index
        this_inst = {
            "input_ids": [],
            "token_type_ids": [],
            "mc_token_ids": [],
            "mlm_label":[]
            }
        
        if self.type=='train':
            if self.args.negative_sample_method == "random":
                #안될경우 다른 문서에서 가져온다.
                # selected_neg_idx=random.sample(neg_cand_idx,self.args.negative_num+1)
                selected_neg_idx=random.sample(range(len(self.all_response)), self.args.negative_num+1)

            
            elif self.args.negative_sample_method == "ranking":
                #random
                if self.negative is None:
                    # selected_neg_idx=random.sample(neg_cand_idx,self.args.negative_num+1)
                    selected_neg_idx=random.sample(range(len(self.all_response)), self.args.negative_num+1)
                else:
                    selected_neg_idx=[self.negative[sample_id][0]]+random.sample(range(len(self.all_response)), self.args.negative_num)
                    #debug
                    if selected_neg_idx is None:
                        # selected_neg_idx=random.sample(neg_cand_idx,self.args.negative_num+1)
                        selected_neg_idx=random.sample(range(len(self.all_response)), self.args.negative_num+1)

            #혹시 랜덤 뽑다가 정답걸릴수도.    
            selected_negative=[self.all_response[idx] for idx in selected_neg_idx]
            if postive_resp in selected_negative:
                selected_negative,selected_neg_idx=self.remove_answer(postive_resp,selected_negative,selected_neg_idx)
            selected_negative=selected_negative[:self.args.negative_num]


            selected_negative.append(postive_resp)
            random.shuffle(selected_negative)
            candidates = selected_negative
            label_idx = candidates.index(postive_resp)
            this_inst["label_idx"] = label_idx
            this_inst["sample_id"]= sample_id

        elif self.type=='val' or self.type=='test':
           
            #debug for toy_train as dev.
            candidates= [postive_resp]+negative_resp_list
            label_idx=0
            this_inst["label_idx"] = label_idx

        
        elif self.type=='ranking':


            # selected_neg_idx=random.sample(range(len(self.all_response)),self.args.candidates_num)
            selected_neg_idx=np.random.choice(neg_cand_idx, self.args.candidates_num, replace=False)
            selected_negative=[self.all_response[idx] for idx in selected_neg_idx]

            if postive_resp in selected_negative:
                selected_negative,selected_neg_idx=self.remove_answer(postive_resp,selected_negative,selected_neg_idx)

            candidates=[postive_resp]+selected_negative
            this_inst["label_idx"] = 0
            this_inst["sample_id"]= sample_id
            #왜넣지? 인덱스 0부터 시작하니까 대응시킬라고

            this_inst["negative_id"]=selected_neg_idx

        for index,cand in enumerate(candidates):
           
          
            instance= self.build_input_from_segments(context,cand)

            this_inst["input_ids"].append(instance["input_ids"])
            this_inst["token_type_ids"].append(instance["token_type_ids"])

            if self.type=='train' and self.args.multi_task:
                    this_inst["mlm_label"].append(instance["mlm_label"])

        return this_inst
        

    #debug
    def build_input_from_segments(self, context, cand):
        """ Build a sequence of input from 2 segments: knowledge and history"""
        instance = {}
    
        # query_id=self.toke 
        # cls + query + sep + passage
        context="[eos]".join(context)
    
      
        input = self.tokenizer.cls_token+context+ self.tokenizer.sep_token+cand
        input_ids=self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(input))
        if len(input_ids)>self.args.max_seq:
            input_ids=[self.tokenizer.cls_token_id]+input_ids[-(self.args.max_seq-1):]
        context_len = input_ids.index(self.tokenizer.sep_token_id)+1


        # context_truncate
        # input = self.tokenizer.cls_token+context+self.tokenizer.sep_token+cand
        # input_ids=self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(input))
        # if len(input_ids)>self.args.max_seq:
        #     middle=input_ids.index(self.tokenizer.sep_token_id)
        #     start_idx=max(0,middle-(self.args.max_seq//2)+1)
        #     input_ids=[self.tokenizer.cls_token_id]+input_ids[start_idx:middle+(self.args.max_seq//2)]
        #     # print("middle %d and length %d"%(middle,len(input_ids)))
        # context_len = input_ids.index(self.tokenizer.sep_token_id)+1


            
        #truncate
       
        # instance[""]
        if self.args.multi_task and self.type=='train':
            sequence, mlm_label=self.random_word(input_ids)
        else:
            sequence= input_ids


        token_type_ids = [0]*context_len+[1]*(len(sequence)-context_len)

        instance["input_ids"] = sequence
        instance["token_type_ids"] = token_type_ids
        if self.args.multi_task and self.type=='train':
            instance["mlm_label"]=mlm_label
        
        return instance
    

    def __len__(self):
        return len(self.examples)

    def remove_answer(self,answer,selected_negative,selected_neg_idx):
        index = selected_negative.index(answer)
        # index=np.where(selected_negative == answer)[0][0]
        selected_negative.pop(index)
        selected_neg_idx = np.delete(selected_neg_idx, index)
        # selected_neg_idx.pop(index)
        return selected_negative, selected_neg_idx
        



    def random_word(self, tokens):
      
        mlm_label = []

        for i, token_id in enumerate(tokens):
            

            if token_id in self.tokenizer.all_special_ids:
                mlm_label.append(-100)
                continue
            
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.mask

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(self.tokenizer.vocab_size)

                # -> rest 10% randomly keep current token

                # append current token to output (we will predict these later)
                try:
                    #debug
                    mlm_label.append(token_id)
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    mlm_label.append(self.unk)
                    
            else:
                # no masking token (will be ignored by loss function later)
                mlm_label.append(-100)

        return tokens, mlm_label

    
    def collate_fn(self, batch):
        input_ids = [ids for ins in batch for ids in ins["input_ids"]]
        token_type_ids = [ids for ins in batch for ids in ins["token_type_ids"]]
        label_idx = [ins["label_idx"] for ins in batch]
        
        if self.args.multi_task and self.type=='train':
            mlm_label = [ids for ins in batch for ids in ins["mlm_label"]]
            mlm_label = torch.tensor(pad_ids(mlm_label,-100))
        if self.type=='ranking':
            data_info={
                "sample_id":[ins["sample_id"] for ins in batch],
                "negative_id":[ins["negative_id"] for ins in batch]
            }


        batch_size = len(batch)
        n_candidates = len(batch[0]["input_ids"])
        
        # if self.type=='sampling':
        #     input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        # else:
        input_ids = torch.tensor(pad_ids(input_ids, self.pad)).view(batch_size, n_candidates, -1)
        
        #MMM [md]
        token_type_pad = token_type_ids[0][0] 
        
        #token_type_pad = self.pad
        
        # if self.type=='sampling':
        #     token_type_ids = torch.tensor(pad_ids(token_type_ids, token_type_pad))
        # else:
        token_type_ids = torch.tensor(pad_ids(token_type_ids, token_type_pad)).view(batch_size, n_candidates, -1)
        
      
        label_idx = torch.tensor(label_idx)
        
        if self.args.multi_task and self.type=='train':
            return input_ids, token_type_ids, label_idx, mlm_label
        elif self.type=='ranking':
            return input_ids, token_type_ids, label_idx, data_info
        else:
            return input_ids, token_type_ids, label_idx