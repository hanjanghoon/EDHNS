from html.entities import entitydefs
import json
from tqdm import tqdm
import re
import numpy as np
import random

# checkpoint_path="runs/dstc9_final/re_rlm_rr_05_512_seed0_rs12_3test_pa10_gpu12_long/checkpoint-399/pred_test_seen.json"
# checkpoint_path="runs/dstc9_final/rlm_rr_05_64_seed0_rs12_3test_pa10_gpu12_long_bm25/checkpoint-1995/pred_test_seen.json"
checkpoint_path="runs/dstc9_test_part/rl_05_gpu8_bm25_200_seed404/checkpoint-2400/pred_test.json"
# checkpoint_path="runs/dstc9_test_part/rl_05_gpu8_bm25_200/checkpoint-6000/pred_test.json"
answer_path='dstc9_data/test/labels.json'
dialog_path='dstc9_data/test/logs.json'

# answer_path='dstc9_data/test_seen_toy/labels.json'
# dialog_path='dstc9_data/test_seen_toy/logs.json'

# checkpoint_path="runs/dstc9_final/re_rlm_rr_05_512_seed0_rs12_3test_pa10_gpu12_long/checkpoint-1596/pred_val.json"
# checkpoint_path="runs/dstc9_final/rlm_rr_05_64_seed0_rs12_3test_pa10_gpu12_long_bm25/checkpoint-798/pred_val.json"
# checkpoint_path="runs/dstc9_final/rlm_rr_05_512_rs12_pa10_gpu12_long_bmen25_toy_c4/checkpoint-1197/pred_val.json"
# answer_path='dstc9_data/val/labels.json'
# dialog_path='dstc9_data/val/logs.json'

knowledge_path='dstc9_data/test/knowledge.json'


with open(checkpoint_path, 'r') as f:
    pred_logs = json.load(f)

with open(dialog_path, 'r') as f:
    dialogs = json.load(f)

with open(answer_path, 'r') as f:
    labels = json.load(f)

with open(knowledge_path, 'r') as f:
    knowledge = json.load(f)


with open('dstc9_data/knowledge.json', 'r') as f:
    knowledge_val = json.load(f)


#dataset 
with open('dstc9_data/train/logs.json', 'r') as f:
    train_logs = json.load(f)

with open('dstc9_data/train/labels.json', 'r') as f:
    train_labels = json.load(f)

with open('dstc9_data/val/logs.json', 'r') as f:
    val_logs = json.load(f)

with open('dstc9_data/val/labels.json', 'r') as f:
    val_labels = json.load(f)

with open('dstc9_data/test_seen/logs.json', 'r') as f:
    test_seen_logs = json.load(f)

with open('dstc9_data/test_seen/labels.json', 'r') as f:
    test_seen_labels = json.load(f)

with open('dstc9_data/test/logs.json', 'r') as f:
    test_logs = json.load(f)

with open('dstc9_data/test/labels.json', 'r') as f:
    test_labels = json.load(f)

with open('dstc9_data/test_unseen_domain/logs.json', 'r') as f:
    test_unseen_domain_logs = json.load(f)

with open('dstc9_data/test_unseen_domain/labels.json', 'r') as f:
    test_unseen_domain_labels = json.load(f)

with open('dstc9_data/test_unseen_entity/logs.json', 'r') as f:
    test_unseen_entity_logs = json.load(f)

with open('dstc9_data/test_unseen_entity/labels.json', 'r') as f:
    test_unseen_entity_labels = json.load(f)


def check_faq(labels):
    faq_list=[]
    for label in labels:
        if label['target']==False:
            continue
        label=label['knowledge'][0]
        domain=label['domain']
        entity_id=label['entity_id']
        doc_id=label['doc_id']
        faq="%s__%s__%s"%(domain,entity_id,doc_id)
        faq_list.append(faq)

    return set(faq_list)



def compare(pred, target, dialogs, knowledge,pure_test):
    
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

    test_seen=[]
    test_unseen_entity=[]
    test_unseen_domain=[]
    for i in range(len(pred)):
        print("index: %d"%i)
        
        
        
        if pred[i]['target']==False:
            continue
        

        true_flag=False
        pred_domain=pred[i]['knowledge'][0]['domain']
        pred_entity_id=pred[i]['knowledge'][0]['entity_id']
        pred_doc_id=pred[i]['knowledge'][0]['doc_id']

        target_domain=target[i]['knowledge'][0]['domain']
        target_entity_id=target[i]['knowledge'][0]['entity_id']
        target_doc_id=target[i]['knowledge'][0]['doc_id']

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

        if True:#true_flag!=True:
          
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
            if target_key not in pure_test:
                print("Seen")
            else:
                print("Unseen")
            print()
            
            print("pred:")
            for j in range(5):
                print(pred[i]['knowledge'][j])
                print(knowledge[pred[i]['knowledge'][j]['domain']][str(pred[i]['knowledge'][j]['entity_id'])]['docs'][str(pred[i]['knowledge'][j]['doc_id'])])
                print(knowledge[pred[i]['knowledge'][j]['domain']][str(pred[i]['knowledge'][j]['entity_id'])]['name'])
                pred_key="%s__%s__%s"%(pred[i]['knowledge'][j]['domain'],pred[i]['knowledge'][j]['entity_id'],pred[i]['knowledge'][j]['doc_id'])
                if pred_key not in pure_test:
                    print("Seen--")
                else:
                    print("Unseen--")
                
            if target_key in test_seen_labels:
                if true_flag==True:
                    test_seen.append(1)
                else:
                    test_seen.append(0)
            elif target_key in test_unseen_domain_labels:
                if true_flag==True:
                    test_unseen_domain.append(1)
                else:
                    test_unseen_domain.append(0)
            elif target_key in test_unseen_entity_labels:
                if true_flag==True:
                    test_unseen_entity.append(1)
                else:
                    test_unseen_entity.append(0)
            best_list=[]
            # best_list=[23, 33, 40, 56, 73, 81, 98, 107, 114, 142, 154, 171, 226, 295, 305, 319, 337, 384, 407, 412, 419, 426, 429, 435, 439, 462, 466, 512, 522, 528, 530, 534, 540, 553, 562, 566, 570, 577, 581, 595, 596, 601, 615, 633, 636, 667, 681, 685, 694, 698, 727, 739, 797, 817, 821, 835, 849, 857, 858, 860, 866, 868, 870, 896, 903, 908, 917, 933, 949, 950, 951, 956, 979, 992, 1015, 1042, 1047, 1054, 1071, 1091, 1107, 1109, 1112, 1125, 1139, 1142, 1199, 1211, 1213, 1227, 1251, 1259, 1276, 1289, 1290, 1311, 1328, 1338, 1375, 1388, 1389, 1436, 1486, 1509, 1539, 1554, 1583, 1603, 1610, 1621, 1658, 1667, 1672, 1731, 1734, 1740, 1767, 1771, 1773, 1823, 1842, 1848, 1885, 1896, 1901, 1903, 1910, 1912, 1913]
            if true_flag!=True and i not in best_list:
                if target[i]['knowledge'][0] in pred[i]['knowledge']:
                    weak_false+=1
                    if target_key not in pure_test:
                        weak_false_seen+=1
                    else:
                        weak_false_unseen+=1
                else:
                    strong_false+=1
                    if target_key not in pure_test:
                        strong_false_seen+=1
                    else:
                        strong_false_unseen+=1
               

           
            print('\n\n\n')
        print("----------") 
    print("total: %d"%len(pred)) 
    print("domain_false: %d"%domain_false)
    print("entity_false: %d"%entity_false)
    print("doc_false: %d"%doc_false)
    print("etc_false: %d"%etc_false)
    print("weak_false: %d"%weak_false)
    print("weak_false_seen: %d"%weak_false_seen)
    print("weak_false_unseen: %d"%weak_false_unseen)
    print("strong_false: %d"%strong_false)
    print("strong_false_seen: %d"%strong_false_seen)
    print("strong_false_unseen: %d"%strong_false_unseen)
    print("test_seen: %.5f"%(sum(test_seen)/len(test_seen)))
    print("test_unseen_entity: %.5f"%(sum(test_unseen_entity)/len(test_unseen_entity)))
    print("test_unseen_domain: %.5f"%(sum(test_unseen_domain)/len(test_unseen_domain)))
    print(incorrect)

def make_seen_toy():
    toy_logs=[]
    toy_labels=[]
    hard_sample=[3, 8, 12, 13, 18, 22, 27, 30, 32, 34, 41, 47, 55, 58, 59, 61, 65, 72, 73, 75, 80, 87, 90, 94, 103, 107, 109, 113, 115, 127, 135, 138, 140, 147, 148, 149, 159, 168, 171, 179, 188, 192, 197, 198, 200, 206, 216, 218, 219, 232, 234, 238, 241, 243, 244, 248, 251, 252, 257, 261, 267, 272, 276, 289, 290, 294, 295, 301, 303, 308, 313, 317, 321, 328, 333, 346, 358, 360, 364, 400, 415, 433, 438, 461, 463, 466, 470, 471, 479, 489, 491, 492, 497, 506, 522, 523, 526, 528, 529, 532, 534, 536, 541, 543, 551, 557, 559, 568, 574, 576, 578, 583, 596, 599, 601, 602, 605, 606, 612, 614, 624, 631, 640, 648, 650, 653, 660, 667, 692, 694, 703, 722, 723, 725, 734, 741, 749, 762, 763, 774, 776, 780, 792, 794, 796, 797, 805, 816, 820, 821, 832, 858, 866, 875, 876, 878, 879, 884, 887, 895, 896, 903, 914, 915, 919, 929, 937, 942, 954, 968, 976]
    print(len(hard_sample))
    for j in hard_sample:
        toy_logs.append(test_seen_logs[j])
        toy_labels.append(test_seen_labels[j])
    with open('dstc9_data/test_seen_toy/logs.json', 'w') as f:
        json.dump(toy_logs, f, indent=4)
    with open('dstc9_data/test_seen_toy/labels.json', 'w') as f:
        json.dump(toy_labels, f, indent=4)
    return


train_labels=check_faq(train_labels)
val_labels=check_faq(val_labels)
# make_seen_toy()

test_labels=check_faq(test_labels)
test_seen_labels=check_faq(test_seen_labels)
test_unseen_domain_labels=check_faq(test_unseen_domain_labels)
test_unseen_entity_labels=check_faq(test_unseen_entity_labels)
pure_val=val_labels-train_labels
pure_test_seen=test_seen_labels-train_labels
pure_test=test_labels-train_labels
compare(pred_logs,labels,dialogs,knowledge,pure_test)
print()