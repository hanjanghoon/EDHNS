from curses import raw
import os
import json
import random
import logging
import sys
import numpy as np
from itertools import chain

import torch

from tqdm import tqdm

from utils.data import (
    pad_ids, truncate_sequences
)
from transformers import RobertaTokenizer, BertTokenizer
from utils.dataset_walker import DatasetWalker
from utils.knowledge_reader import KnowledgeReader
from rank_bm25 import BM25Okapi


logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>","<eou>","<mask>"],
}
SPECIAL_TOKENS_VALUES = ["<bos>", "<eos>", "<pad>", "<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>","<eou>","<mask>"]

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        self.args = args
        self.dataroot = args.dataroot
        self.tokenizer = tokenizer
        self.split_type = split_type

        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
        self.bos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
        self.eos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])
        self.pad = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])
        self.speaker1, self.speaker2, self.knowledge_sep, self.knowledge_tag, self.eou_tag, self.mask_tag  = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["additional_special_tokens"]
        )
        self.knowledge_sep_token = self.SPECIAL_TOKENS["additional_special_tokens"][2]

        self.dataset_walker = DatasetWalker(split_type, labels=labels, dataroot=self.dataroot, labels_file=labels_file)
        

        self.knowledge_reader = KnowledgeReader(self.dataroot, args.knowledge_file)
        
        self.knowledge, raw_snippets, self.snippets, self.entity_name2key = self._prepare_knowledge()
        self.dialogs = self._prepare_conversations()


        if args.negative_sample_method=='scale' and "dstc10" in args.dataroot and split_type=='train':
            with open('smil_knowledge.json', 'r') as f:
                self.sbert_key = json.load(f)
        
        if (split_type=='train') and args.negative_sample_method=='bm25' :
            with open(self.dataroot+'/bm25_negative45.json', 'r') as f:
                confined_candidates = json.load(f)
            self.bm25_cand=confined_candidates['bm25']


        self._create_examples()


    def _knowledge_to_string(self, selection_type, doc, name="", domain=""):
        join_str = " %s " % self.knowledge_sep_token
        if selection_type == "all":
            return join_str.join([domain, name, doc["title"], doc["body"]])


    def _prepare_conversations(self):
        logger.info("Tokenize and encode the dialog data")
        tokenized_dialogs = []
        for i, (log, label) in enumerate(tqdm(self.dataset_walker, disable=self.args.local_rank not in [-1, 0],desc=self.split_type)): # only show progress bar in one process
            # #jang for debug
            # if self.split_type =='train' or self.split_type =='ranking' :
            #     if i>=300:
            #         break
                
            # if 'test' in self.split_type :
            #     if i>=5:
            #         break
            
            # if 'val' in self.split_type :
            #     if i>=10:
            #         break
                
            
            dialog = {}
            dialog["id"] = i
            dialog["log"] = log
            if label is not None:
                if "response" in label:
                    label["response_tokenized"] = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(label["response"].lower())
                    )
            dialog["label"] = label
            tokenized_dialogs.append(dialog)
        return tokenized_dialogs
    
    def _prepare_knowledge(self):
        self.knowledge_docs = self.knowledge_reader.get_doc_list()

        tokenized_snippets = dict()
        raw_snippets=dict()
        entity_name2key=dict()
        max_faq_len=0
        for snippet in self.knowledge_docs:
            key = "{}__{}__{}".format(snippet["domain"], str(snippet["entity_id"]) or "", snippet["doc_id"])
            
            #for cross_entity
            if snippet['entity_name'] is None:
                entity_name=snippet['domain']
            else:
                entity_name=snippet['entity_name']
            
            if entity_name not in list(entity_name2key.keys()):
                entity_name2key[entity_name]=[]
            entity_name2key[entity_name].append(key)
            
            #for training
            knowledge_all = self._knowledge_to_string('all', snippet["doc"], name=snippet["entity_name"] or "", domain=snippet["domain"] or "")
            knowledge_all=knowledge_all.lower()
            raw_snippets[key]=knowledge_all
            tokenized_knowledge = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(knowledge_all))
            if len(tokenized_knowledge)>max_faq_len and self.args.local_rank in [-1, 0]:
                max_faq_len=len(tokenized_knowledge)
            tokenized_snippets[key] = tokenized_knowledge[:self.args.knowledge_max_tokens]
        
        
        if self.args.local_rank in [-1, 0] :
            print("max_faq_length: %d"%max_faq_len)
        return knowledge_all, raw_snippets, tokenized_snippets,entity_name2key

    def find_cross_entity(self,dialog,entity_name):
        in_cross_entity=[]
        for uttr in dialog:
            for name in list(self.entity_name2key.keys()):
                # if name is None:
                #     name=
                if name.lower() in uttr['text'].lower() and name.lower()!=entity_name.lower():
                    in_cross_entity.extend(self.entity_name2key[name])
        
        return list(set(in_cross_entity))
        
    def _create_examples(self):
        max_context_len=0
        
        logger.info("Creating examples")
        self.examples = []
        for dialog in tqdm(self.dialogs, disable=self.args.local_rank not in [-1, 0]):
            label = dialog["label"]
            dialog = dialog["log"]
            if label is None:
                label = {"target": False}

            target = label["target"]

            if not target:
                continue
                    
            history = [
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(turn["text"].lower()))
                for turn in dialog
            ]
            history_len = sum([len(tokenized_turn) for tokenized_turn in history])
            if self.args.local_rank in [-1, 0] and max_context_len<history_len:
                max_context_len=history_len

            # perform token-level truncation of history from the left 앞에서 부터 자른다 잘짜졌음.
            truncated_history = truncate_sequences(history, self.args.history_max_tokens)
            #jang
            knowledge_key=None
            if target:
                if "knowledge" not in label:
                    label["knowledge"] = [self.knowledge_docs[0]]
                knowledge = label["knowledge"][0]
                knowledge_key = "{}__{}__{}".format(knowledge["domain"], knowledge["entity_id"], knowledge["doc_id"])

                
                entity_name=self.knowledge_reader.get_entity_name(knowledge["domain"],knowledge["entity_id"])
                if entity_name is None:
                    entity_name=knowledge["domain"]


                prefix_entity = "{}__{}".format(knowledge["domain"], knowledge["entity_id"])
                
                
                in_entitiy_candidates = [
                    cand
                    for cand in self.snippets.keys() 
                    if "__".join(cand.split("__")[:-1]) == prefix_entity
                ]
                in_domain_candidates = [
                    cand
                    for cand in self.snippets.keys() 
                    if cand.split("__")[0] == knowledge["domain"]
                ]
                cross_entity_candidates=self.find_cross_entity(dialog,entity_name)
                
                #jang 여기
                
                # if self.split_type == "train" and self.args.negative_sample_method == "oracle":
                #     # if there's not enough candidates during training, we just skip this example
                #     if len(knowledge_candidates) < self.args.n_candidates:
                #         continue

                used_knowledge = self.snippets[knowledge_key]
                used_knowledge = used_knowledge[:self.args.knowledge_max_tokens]#이거 body만
            
            else:
                in_entitiy_candidates = None
                used_knowledge = []
         
            self.examples.append({
                "history": truncated_history,
                "knowledge": used_knowledge,
                "in_entity": in_entitiy_candidates,
                "in_domain":in_domain_candidates,
                "cross_entity":cross_entity_candidates,
                "knowledge_key":knowledge_key,
                "label": label,
                "knowledge_seeking": target
            })
        if self.args.local_rank in [-1, 0] :
            print("max_context_length in %s: %d"%(self.split_type,max_context_len))
                
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.examples)


#jang-r
class KnowledgeSelectionDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(KnowledgeSelectionDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)
        
        #entity,domain,relevance,random
        #이전 논문은 1:2:2:2 였음
        # self.weight=[0.14,0.28,0.28,0.3]
        # self.weight=[0,0,0,1.0]
        self.negative=None
     


        
    # def _split_int_array(self, seq, smallest):    
    #     group = []    
    #     for num in seq:
    #         if num != smallest:
    #             group.append(num)
    #         elif group:
    #             yield group
    #             group = []
    #     if group:
    #         yield group
    
  


    def __getitem__(self, index):
        example = self.examples[index]

        this_inst = {
            "dialog_id": index,
            "input_ids": [],
            "token_type_ids": [],
            "mc_token_ids": [],
            "mlm_label":[]
        }
        

        if self.split_type == "val" :
            candidate_keys = list(self.snippets.keys())
            candidates = [self.snippets[cand_key] for cand_key in candidate_keys]

        elif "test" in self.split_type:
            # if False:
            if self.args.eval_only:
                #test_whole
                candidate_keys = list(self.snippets.keys())
            
            else:
                if "dstc10" in self.dataroot:
                    candidate_keys = list(self.snippets.keys())
                else:
                    # test_part
                    knowledge_key=knowledge_key=example["knowledge_key"]  
                    candidate_keys=random.sample(list(self.snippets.keys()),min(2048, len(list(self.snippets.keys()) )))
                    while knowledge_key in candidate_keys:
                        candidate_keys.remove(knowledge_key)

                    candidate_keys=[knowledge_key]+candidate_keys
                    random.shuffle(candidate_keys)


            candidates = [self.snippets[cand_key] for cand_key in candidate_keys]

        elif self.split_type=='train':
            knowledge_key=example["knowledge_key"]
            
            if self.args.negative_sample_method == "random":
                pool_keys = list(self.snippets.keys())
                negative_keys=[]
                while negative_keys <=self.args.negaitve_num:
                    key=random.sample(pool_keys,1)[0]
                    if key!=knowledge_key:
                        negative_keys.append(key)
            
            elif self.args.negative_sample_method == "scale":
                negative_keys=[]
                while len(negative_keys) <=self.args.negative_num:
                    
                    if "dstc10" in self.args.dataroot:
                        pool=random.sample(['random','in_entity','in_domain','cross_entity','relative'],1)[0]
                    else:
                        pool=random.sample(['random','in_entity','in_domain','cross_entity'],1)[0]

                    if pool=="random":
                        pool_keys=list(self.snippets.keys())
                    elif pool=='in_entity':
                        pool_keys=example['in_entity']
                    elif pool=='in_domain':
                        pool_keys=example['in_domain']
                    elif pool=='cross_entity':
                        pool_keys=example['cross_entity']
                    elif pool=='relative' and "dstc10" in self.args.dataroot:
                        pool_keys=self.sbert_key[knowledge_key]
                    
                    if pool_keys:
                        key=random.sample(pool_keys,1)[0]
                        if key!=knowledge_key:
                            negative_keys.append(key)
            
            elif self.args.negative_sample_method == "bm25":
                pool_keys=self.bm25_cand[str(index)]
                negative_keys=random.sample(pool_keys,self.args.negative_num)
                
            candidate_keys=negative_keys+[knowledge_key]
            random.shuffle(candidate_keys)
            candidates = [self.snippets[cand_key] for cand_key in candidate_keys]
            
        
        
        this_inst["candidate_keys"] = candidate_keys
        
        label_idx = candidates.index(example["knowledge"])
        this_inst["label_idx"] = label_idx

        for cand in candidates:
            instance, _ = self.build_input_from_segments(
                cand,
                example["history"]
            )
            this_inst["input_ids"].append(instance["input_ids"])
            this_inst["token_type_ids"].append(instance["token_type_ids"])
            if self.args.multi_task and self.split_type=='train':
                this_inst["mlm_label"].append(instance["mlm_label"])

        return this_inst

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
                    tokens[i] = self.mask_tag

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(self.tokenizer.vocab_size)

                # -> rest 10% randomly keep current token

                # append current token to output (we will predict these later)
                try:
                    mlm_label.append(token_id)
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    mlm_label.append(self.tokenizer.unk_token_id)
                    logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(tokens))
            else:
                # no masking token (will be ignored by loss function later)
                mlm_label.append(-100)

        return tokens, mlm_label


    def build_input_from_segments(self, knowledge, history):
        """ Build a sequence of input from 2 segments: knowledge and history"""
        instance = {}
        context=[self.bos] 
        for u in history:
            context+=u+[self.eou_tag]
        context+=[self.knowledge_tag]#sep

        context=context[-self.args.history_max_tokens:]
        
        sequence = context + knowledge + [self.eos]
        context_len=len(context)

        
        if self.args.multi_task and self.split_type=='train':
            sequence, mlm_label=self.random_word(sequence)

        # token_type_ids = [0]*context_len+[1]*(len(sequence)-context_len)
        
        # roberta and deberta
        if self.args.premodel=="roberta" or self.args.premodel=="deberta":
            token_type_ids = [0]*len(sequence)
        #bert
        else:
            token_type_ids=[0]*context_len+[1]*(len(sequence)-context_len)

        instance["input_ids"] = sequence
        instance["token_type_ids"] = token_type_ids
        
        if self.args.multi_task and self.split_type=='train':
            instance["mlm_label"]=mlm_label

        return instance, sequence
    


    def collate_fn(self, batch):
        input_ids = [ids for ins in batch for ids in ins["input_ids"]]
        token_type_ids = [ids for ins in batch for ids in ins["token_type_ids"]]
        label_idx = [ins["label_idx"] for ins in batch]
        
        if self.args.multi_task and self.split_type=='train':
            mlm_label = [ids for ins in batch for ids in ins["mlm_label"]]
            mlm_label = torch.tensor(pad_ids(mlm_label,-100))

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch],
            "candidate_keys": [ins["candidate_keys"] for ins in batch]
        }

        batch_size = len(batch)
        n_candidates = len(batch[0]["input_ids"])
        
        input_ids = torch.tensor(pad_ids(input_ids, self.pad)).view(batch_size, n_candidates, -1)
        token_type_pad = token_type_ids[0][0] 
        token_type_ids = torch.tensor(pad_ids(token_type_ids, token_type_pad)).view(batch_size, n_candidates, -1)
        label_idx = torch.tensor(label_idx)
        
        if self.args.multi_task and self.split_type=='train':
            return input_ids, token_type_ids, label_idx, mlm_label, data_info
        else:
            return input_ids, token_type_ids, label_idx, data_info
