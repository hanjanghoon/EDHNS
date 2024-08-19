from html.entities import entitydefs
import json
from tqdm import tqdm
import re
import numpy as np
import random

# checkpoint_path="runs/dstc10/rl_05_64_gpu4_c2_rs120_new_data/checkpoint-9350/pred_test_train.json"
# answer_path='dstc10_data/test_train/labels.json'
# dialog_path='dstc10_data/test_train/logs.json'


# checkpoint_path="runs/dstc10_knover/rlm_05_64_gpu8_c2_rs120_kntrim_testall/checkpoint-20068/pred_test_part.json"
# answer_path='dstc10_data/test/labels.json'
# dialog_path='dstc10_data/test/logs.json'



checkpoint_path="runs/dstc10_sample/rlm_05_20_sbert_228_gpu8_rs32/checkpoint-2635/pred_val.json"
answer_path='dstc10_data_sample/val/labels.json'
dialog_path='dstc10_data_sample/val/logs.json'

knowledge_path='dstc10_data_sample/knowledge.json'


with open(checkpoint_path, 'r') as f:
    pred_logs = json.load(f)

with open(dialog_path, 'r') as f:
    dialogs = json.load(f)

with open(answer_path, 'r') as f:
    labels = json.load(f)

with open(knowledge_path, 'r') as f:
    knowledge = json.load(f)


#dataset 
with open('dstc10_data_sample/train/logs.json', 'r') as f:
    train_logs = json.load(f)

with open('dstc10_data_sample/train/labels.json', 'r') as f:
    train_labels = json.load(f)

with open('dstc10_data_sample/val/logs.json', 'r') as f:
    val_logs = json.load(f)

with open('dstc10_data_sample/val/labels.json', 'r') as f:
    val_labels = json.load(f)

with open('dstc10_data_sample/test/logs.json', 'r') as f:
    test_logs = json.load(f)

with open('dstc10_data_sample/test/labels.json', 'r') as f:
    test_labels = json.load(f)



def check_faq(labels):
    faq_list=[]
    faq_dict={}
    entity_id_dict={}
    for label in labels:
        if label['target']==False:
            continue
        label=label['knowledge'][0]
        domain=str(label['domain'])
        entity_id=str(label['entity_id'])
        doc_id=str(label['doc_id'])
        faq="%s__%s__%s"%(domain,entity_id,doc_id)
        faq_list.append(faq)
        if entity_id not in entity_id_dict:
            entity_id_dict[entity_id]=0
        entity_id_dict[entity_id]+=1
        if faq not in faq_dict:
            faq_dict[faq]=0
        faq_dict[faq]+=1

    return set(faq_list), faq_dict,entity_id_dict



def compare(pred, target, dialogs, knowledge,pure_eval):
    
    domain_false=0
    entity_false=0
    doc_false=0
    etc_false=0
    weak_false, weak_false_seen,weak_false_unseen=0,0,0
    strong_false, strong_false_seen,strong_false_unseen=0,0,0

    
    last_k=-1000

    incorrect=[]

    #오직 detection true 만

    valid_target=[]
    valid_dialog=[]
    for i in range(len(target)):
        if target[i]['target']==True:
            valid_target.append(target[i])
            valid_dialog.append(dialogs[i])
    
    target=valid_target
    dialogs=valid_dialog

    total_cnt=0
    for i in range(len(pred)):
        # print("index: %d"%i)
        
        
        
        if pred[i]['target']==False:
            continue
        total_cnt+=1

        true_flag=False
        pred_domain=pred[i]['knowledge'][0]['domain']
        pred_entity_id=str(pred[i]['knowledge'][0]['entity_id'])
        pred_doc_id=str(pred[i]['knowledge'][0]['doc_id'])

        target_domain=target[i]['knowledge'][0]['domain']
        target_entity_id=str(target[i]['knowledge'][0]['entity_id'])
        target_doc_id=str(target[i]['knowledge'][0]['doc_id'])

        target_key="%s__%s__%s"%(target_domain,target_entity_id,target_doc_id)

        if (pred_domain==target_domain) and (pred_entity_id == target_entity_id) and (pred_doc_id == target_doc_id):
            true_flag=True
            print("True!")
        # elif pred_domain != target_domain:
        #     print("domain x\n")
        #     domain_false+=1
        elif pred_entity_id != target_entity_id:
            print("entity_id x\n")
            entity_false+=1
        elif pred_doc_id != target_doc_id:
            print("doc_id x\n")
            doc_false+=1
        else:
            print("what?\n")
            etc_false+=1
        
        if true_flag!=True:
            incorrect.append(i)

        if true_flag!=True:
          
            target_faq=knowledge[target_domain][str(target_entity_id)]['docs'][str(target_doc_id)]
            target_name=knowledge[target_domain][str(target_entity_id)]['name']
            pred_faq=knowledge[pred_domain][str(pred_entity_id)]['docs'][str(pred_doc_id)]
            
            dialog=[]
            for k,uttr in enumerate(dialogs[i]):
                dialog.append(uttr['text'])
                if target_name:
                    if target_name.lower() in uttr['text'].lower():
                        last_k=k
            gap=len(dialog)-last_k
            question=dialog[-1]

            print("qeustion:")
            print(question)
            print()

            print("target:")
            print(target[i]['knowledge'][0])
            print(target_faq)
            print(target_name)
       
            print()
            
            print("pred:")
            for j in range(5):
                print(pred[i]['knowledge'][j])
                print(knowledge[pred[i]['knowledge'][j]['domain']][str(pred[i]['knowledge'][j]['entity_id'])]['docs'][str(pred[i]['knowledge'][j]['doc_id'])])
                print(knowledge[pred[i]['knowledge'][j]['domain']][str(pred[i]['knowledge'][j]['entity_id'])]['name'])
                pred_key="%s__%s__%s"%(pred[i]['knowledge'][j]['domain'],pred[i]['knowledge'][j]['entity_id'],pred[i]['knowledge'][j]['doc_id'])
                
            target[i]['knowledge'][0]={'domain':target_domain,'entity_id':int(target_entity_id),'doc_id':int(target_doc_id)}

            # if i in [1, 6, 16, 31, 32, 35, 42, 63, 65, 68, 72, 75, 81, 101, 116, 129, 136, 138, 154, 165, 166, 170, 179, 184, 185, 200, 220, 224, 226, 228, 235, 238, 251, 257, 265, 271, 280, 289, 303, 321, 322, 324, 331, 340, 341, 345, 348, 352, 355, 356, 364, 376, 379, 383, 388, 393, 401, 409, 421, 437, 450, 456, 467, 474, 476, 511, 512, 514, 518, 528, 533, 544, 559, 567, 574, 575, 576, 581, 617, 632, 639, 647, 654, 664, 668, 669, 675, 681]:
                # continue

            if true_flag!=True:
                if target[i]['knowledge'][0] in pred[i]['knowledge']:
                    weak_false+=1
                    if target_key not in pure_eval:
                        weak_false_seen+=1
                    else:
                        weak_false_unseen+=1
                else:
                    strong_false+=1
                    if target_key not in pure_eval:
                        strong_false_seen+=1
                    else:
                        strong_false_unseen+=1

            # if true_flag!=True:
            #     if target[i]['knowledge'][0] in pred[i]['knowledge']:
            #         weak_false_seen+=1
                    
            #     else:
            #         strong_false_seen+=1
                    
                     

            else:
                if true_flag==True:
                    print()
            print('\n\n\n')
        print("----------") 


    print("total: %d"%total_cnt) 
    print("domain_false: %d"%domain_false)
    print("entity_false: %d"%entity_false)
    print("doc_false: %d"%doc_false)
    print("etc_false: %d"%etc_false)
    print("weak_false_seen: %d"%weak_false_seen)
    print("weak_false_unseen: %d"%weak_false_unseen)
    print("strong_false_seen: %d"%strong_false_seen)
    print("strong_false_unseen: %d"%strong_false_unseen)
    
 
    
    print(incorrect)

def compare2(target, dialogs, knowledge):
    

    last_k=-1000
    over=0
    under=0
    incorrect=[]

    #오직 detection true 만

    valid_target=[]
    valid_dialog=[]
    for i in range(len(target)):
        if target[i]['target']==True:
            valid_target.append(target[i])
            valid_dialog.append(dialogs[i])
    
    target=valid_target
    dialogs=valid_dialog

    total_cnt=0
    for i in range(len(target)):
        # print("index: %d"%i)
        
        
        
        if target[i]['target']==False:
            continue
        total_cnt+=1

        true_flag=False
      
        target_domain=target[i]['knowledge'][0]['domain']
        target_entity_id=target[i]['knowledge'][0]['entity_id']
        target_doc_id=target[i]['knowledge'][0]['doc_id']

        target_key="%s__%s__%s"%(target_domain,target_entity_id,target_doc_id)

        
        if true_flag!=True:
            incorrect.append(i)

        if True:#true_flag!=True:
          
            target_faq=knowledge[target_domain][str(target_entity_id)]['docs'][str(target_doc_id)]
            target_name=knowledge[target_domain][str(target_entity_id)]['name']
            
            dialog=[]
            for k,uttr in enumerate(dialogs[i]):
                dialog.append(uttr['text'])
                if target_name:
                    if target_name.lower() in uttr['text'].lower():
                        last_k=k
            gap=len(dialog)-last_k
            question=dialog[-1]

            if gap>=20:
                over+=1
                print("qeustion:")
                print(question)
                print()

                print("target:")
                print(target[i]['knowledge'][0])
                print(target_faq)
                print(target_name)
                print('\n\n\n')
                print("----------") 
            else:
                under+=1
    print("total: %d"%total_cnt) 
    print("over vs under: %d vs %d"%(over,under))
 

# compare2(train_labels,train_logs,knowledge)
train_labels, train_faq, train_entity_ids=check_faq(train_labels)
# val_labels=check_faq(val_labels)
# test_labels=check_faq(test_labels)
eval_labels, eval_faq, eval_entity_ids=check_faq(labels)
pure_eval=eval_labels-train_labels
# pure_val=val_labels-train_labels
# pure_test=test_labels-train_labels
compare(pred_logs,labels,dialogs,knowledge,pure_eval)
print()