
import random
from tqdm import tqdm
import pickle
from sentence_transformers import SentenceTransformer,util
import argparse
import faiss
import numpy as np
import os
import parmap
from multiprocessing import Manager
import gc
import json
import time

#debug
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="15"
import multiprocessing
from functools import partial
from tqdm.contrib.concurrent import process_map 



random.seed(0)
np.random.seed(0)

def make_dict(data_file):
    dialog_list=[]
    context_list=[]
    response_list=[]
    all_response=set()
    with open(data_file, 'r') as file:
    # 파일의 모든 줄을 리스트로 읽기
        data = file.readlines()


    # lines=f.readlines()
    for i, triple in enumerate(tqdm(data)):
        triple=triple.strip('\n')
        triple=triple.split("\t")
        if triple[0]=='0':
            response=triple[-1]
            all_response.add(response)
        else:
            
            context=triple[1:-1]
            response=triple[-1]
        
            context=".\n".join(context)
            dialog_list.append([context,response])
            context_list.append(context)
            response_list.append(response)
            all_response.add(response)
    
    
    print("making dialog_dict complete!!!!!!!!!")
    return dialog_list,context_list,response_list,list(all_response)



# context_list,response_list,all_response,I=None,None,None,None
# batch_context_list,batch_response_list,batch_I=None,None,None




def search_in_subset(all_response, query):
    # 부분 집합 데이터
    subset_indices=random.sample(range(len(all_response)), 1000)
    subset_response = all_response[subset_indices]
    candidate_length=len(subset_indices)


    # 부분 집합에 대한 인덱스 생성 및 추가
    index_cpu = faiss.IndexFlatIP(subset_response.shape[-1])
    # index_gpu= faiss.index_cpu_to_gpu(res, 0, index_cpu)
    index_gpu=index_cpu
    index_gpu.add(subset_response)

    # 부분 집합에서 검색
    L,D,I =index_gpu.range_search(query[np.newaxis,:], thresh=0.1)

    start = L[0]
    end = L[1]
    length=end-start
   
    if length<100:
        compensate_I=I[start:end].tolist()+random.sample(range(candidate_length),100)
        compensate_I=list(set(compensate_I))
        ori_I = [subset_indices[i] for i in compensate_I]
    else:
        ori_I = [subset_indices[i] for i in I[start:end].tolist()]
    real_D=D[start:end]

    return real_D, ori_I, len(ori_I)

def main():
    parser = argparse.ArgumentParser()
   
    parser.add_argument("--in_file", type=str, default="e-com/data/train.txt",help="Path to dataset.")
    parser.add_argument("--out_dir", type=str, default="e-com/data/",help="Path to dataset.")
    parser.add_argument('--batch_size', type=int, default= 512, help='batch_size')
    args = parser.parse_args()
    # Setup CUDA, GPU & distributed training
  
    model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
    # model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    #debug
    # model = SentenceTransformer("paraphrase-albert-small-v2")
    
    
    sbert_cand=[]
    
    
    _,context_list,response_list,all_response=make_dict(args.in_file)
    
    print("# of dialog is %d"%len(context_list))
    #debug
    # context_list=context_list[:10000]
    # all_response=all_response[:10000]

    dialog_list=list(zip(context_list,response_list))
    random.shuffle(dialog_list)
    context_list,response_list=zip(*dialog_list)

    print("SBERT_encoding START")
    start = time.time()
    pool = model.start_multi_process_pool()
    context_list_emb = model.encode_multi_process(context_list, pool, batch_size=args.batch_size,chunk_size=100000)
    all_response_emb = model.encode_multi_process(all_response, pool, batch_size=args.batch_size,chunk_size=100000)
    model.stop_multi_process_pool(pool)
    end = time.time()
    print("SBERT_encoding END")
    print(f"{end - start:.2f} sec")


    print("FAISS SEARCH START")
    faiss.normalize_L2(context_list_emb)
    faiss.normalize_L2(all_response_emb)


    D_list = []  # 거리들을 저장할 리스트
    I_list = []  # 인덱스들을 저장할 리스트

    data_num=context_list_emb.shape[0]
    average=0
    for i in tqdm(range(data_num)):
        queries = context_list_emb[i]
        D, original_indices,length=search_in_subset(all_response_emb,queries)
        # 결과를 리스트에 추가합니다.
        average+=length
        D_list.append(D)
        I_list.append(original_indices)
    
    average=average/data_num
    print("all_average=%.2f"%average)
    
    negative_candidate_index = I_list
    print("FAISS SEARCH END")
    results=[]
    for i in range(data_num):
        results.append([context_list[i].split(".\n"),response_list[i],negative_candidate_index[i]])


    with open(args.out_dir+"train_SNDS_1000_01.jsonl" , encoding= "utf-8", mode="w") as jsonl_file:
        for [context, pos, neg_list] in results:
            # if cnt>10000:
            #     break
            dict_data = {"context":context,"pos":pos,"neg_list":neg_list}
            jsonl_file.write(json.dumps(dict_data, ensure_ascii=False) + "\n")
            


    with open(args.out_dir+"train_res_1000_01.pkl", 'wb') as f:
        pickle.dump(all_response, f)

    print("DONE")


 

if __name__ == "__main__":
    main()