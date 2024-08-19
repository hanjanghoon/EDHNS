import json
from numpy.core.arrayprint import set_string_function
from tqdm import tqdm
import re
import numpy as np
import random
import copy
from sentence_transformers import SentenceTransformer, util
from utils.knowledge_reader import KnowledgeReader
import torch
import os
from rank_bm25 import BM25Okapi
random.seed(0)
np.random.seed(0)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="13"

#hotel,restaurant,attraction
#issue! train, texi를 날릴것인가? 일단 도메인이 다르기 때문에 날린다.
# with open('dstc9_data/knowledge.json', 'r') as f:
#     kb9 = json.load(f)

# with open('dstc10_data_sampe/knowledge.json', 'r') as f:
#     kb10 = json.load(f)
  
def _knowledge_to_string(selection_type, doc, name="", domain=""):
    join_str = ", "
    
    if selection_type == "all":
        return join_str.join([domain, name, doc["title"], doc["body"]])

def _prepare_knowledge(knowledge_reader):
    knowledge_docs = knowledge_reader.get_doc_list()
    
  
    raw_snippets=dict()
    entity_id_list=[]
    for snippet in knowledge_docs:
        key = "{}__{}__{}".format(snippet["domain"], str(snippet["entity_id"]) or "", snippet["doc_id"])
        
        if str(snippet["entity_id"]) not in entity_id_list:
            entity_id_list.append(str(snippet["entity_id"]))
        #for training
        knowledge_all = _knowledge_to_string('all', snippet["doc"], name=snippet["entity_name"] or "", domain=snippet["domain"] or "")
        knowledge_all= knowledge_all.lower()
        raw_snippets[key]=knowledge_all

    return raw_snippets


def process_training_data(data_root,raw_snippets):
    with open(data_root+'/train/logs.json', 'r') as f:
        train_logs = json.load(f)
    with open(data_root+'/train/labels.json', 'r') as f:
        train_labels = json.load(f)
    
    context_faq_data=[]
    faq_value=[]
    faq_key=[]
    
    for logs,labels in zip(train_logs,train_labels):
        log_str=""
        
        knowledge="%s__%s__%s"%(labels['knowledge'][0]['domain'],labels['knowledge'][0]['entity_id'],labels['knowledge'][0]['doc_id'])
        faq=raw_snippets[knowledge]
        log_str+=faq+", "

        for utterance in logs:
            log_str+=utterance['text']+", "
        log_str=log_str[:-2]
        
        context_faq_data.append(log_str)
        faq_value.append(faq)
        faq_key.append(knowledge)
    return context_faq_data,faq_value,faq_key

def bert_smilarity(context_faq_data,faq_value,faq_key,raw_snippets,data_type):
    #세번째 dstc10 to dstc10
    sbert_key={}
    sbert_value={}
    model = SentenceTransformer("paraphrase-distilroberta-base-v2")
       #Encode all sentences
    sentences=list(raw_snippets.values())
    embeddings = model.encode(sentences)
    
    average_length=[]
    cnt=0
    for i,(cf,f_key) in enumerate(zip(tqdm(context_faq_data),faq_key)):
        cf_emb = model.encode(cf)
        #Compute cosine similarity between all pairs
        cos_sim = util.cos_sim(cf_emb, embeddings)[0]
        cos_sim_sort = torch.argsort(cos_sim,descending=True,dim=-1)

        f_arg_max= list(raw_snippets.keys()).index(f_key)

        if cos_sim_sort[0]!=f_arg_max:
            cnt+=1

        key_list=[]
        value_list=[]
        for j in cos_sim_sort[1:]:
            # #ver3
            # #자기자신 제거
            if j==f_arg_max:
                continue
            #false negative 제거
            # if f_cos_sim[j]>=0.96:
            #     continue
            #기본 0.45임
            if data_type=='dstc10':
                if (cos_sim[j]<=0.45 and len(key_list)>50) or len(key_list)>=1000:
                    break
            else:
                if (cos_sim[j]<=0.45 and len(key_list)>50) or len(key_list)>=1000:
                    break
            key_list.append(list(raw_snippets.keys())[j])
            value_list.append(list(raw_snippets.values())[j])
        sbert_key[i]=key_list
        sbert_value[i]=value_list
        average_length.append(len(key_list))

    print("average length= %f"%(sum(average_length)/len(average_length)))
    print(cnt)

    return sbert_key,sbert_value

def bm25_smilarity(context_faq_data,faq_value,faq_key,raw_snippets,top_n=1000):
    bm25_keys=dict()
    bm25_contents=dict()
    keys=list(raw_snippets.keys())
    corpus=list(raw_snippets.values())
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    average_length=[]
    cnt=0
    for i,(cf,f_key) in enumerate(zip(tqdm(context_faq_data),faq_key)):
        tokenized_query=  cf.split(" ")
        scores = bm25.get_scores(tokenized_query)
        top_n_index = np.argsort(scores)[::-1][:top_n]
        
        answer_index=keys.index(f_key)
        if top_n_index[0]!=answer_index:
            cnt+=1

        key_list=[]
        value_list=[]
        for j in top_n_index:
            if j==answer_index:
                continue
            
            if scores[j]<=45 and len(key_list)> 50:
                break
            key_list.append(keys[j])
            value_list.append(corpus[j])
        bm25_keys[i]= key_list
        bm25_contents[i]=value_list#자기자신은 스킵
        average_length.append(len(key_list))
    
    print("average length= %f"%(sum(average_length)/len(average_length)))
    print(cnt)
    return bm25_keys,bm25_contents

data_root="dstc10_data_sample_wtest"
knowledge_reader = KnowledgeReader(data_root,"knowledge.json")
raw_snippets= _prepare_knowledge(knowledge_reader)
context_faq_data,faq_value,faq_key=process_training_data(data_root,raw_snippets)
sbert_key,sbert_value=bert_smilarity(context_faq_data,faq_value,faq_key,raw_snippets,'dstc10')
# bm25_key,bm25_value=bm25_smilarity(context_faq_data,faq_value,faq_key,raw_snippets)
# bm25_key=None
# print()


# with open(data_root+'/confined_negative3.json', 'w') as f:
#     json.dump({"sbert":sbert_key,"bm25":bm25_key}, f, indent=4)

with open(data_root+'/sbert_negative045.json', 'w') as f:
    json.dump({"sbert":sbert_key}, f, indent=4)

# with open(data_root+'/bm25_negative45.json', 'w') as f:
#     json.dump({"bm25":bm25_key}, f, indent=4)



print("END Program")

print("end")