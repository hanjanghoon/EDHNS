import json
from tqdm import tqdm
import re
import numpy as np
import random

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

with open('dstc9_data/test_unseen_domain/logs.json', 'r') as f:
    test_unseen_domain_logs = json.load(f)

with open('dstc9_data/test_unseen_domain/labels.json', 'r') as f:
    test_unseen_domain_labels = json.load(f)

with open('dstc9_data/test_unseen_entity/logs.json', 'r') as f:
    test_unseen_entity_logs = json.load(f)

with open('dstc9_data/test_unseen_entity/labels.json', 'r') as f:
    test_unseen_entity_labels = json.load(f)

# with open('dstc9_data/knowledge.json', 'r') as f:
#     kb9_val = json.load(f)
    
with open('dstc9_data/test/knowledge.json', 'r') as f:
    kb9_test = json.load(f)

def check_question_answer(logs,labels):
    QA_list=[]
    entity_list=[]
    entry_list=[]
    for i,log in enumerate(logs):
        log=log[-1]['text']
        if labels[i]['target']==False:
            continue
        domain=labels[i]['knowledge'][0]['domain']
        entity_id=labels[i]['knowledge'][0]['entity_id']
        doc_id=labels[i]['knowledge'][0]['doc_id']
        label=kb9_test[domain][str(entity_id)]['docs'][str(doc_id)]
        QA_list.append((log,label))
        
        entity_name=kb9_test[domain][str(entity_id)]['name']
        if entity_name is None:
            entity_name="*"

        faq=" ".join(label.values())
        entry=entity_name+" "+faq
        if entity_name not in entity_list:
            entity_list.append(entity_name)
        if entry not in entry_list:
            entry_list.append(entry)
    return QA_list,set(entity_list),set(entry_list)

train_set,train_entity,train_entry=check_question_answer(train_logs,train_labels)
val_set,val_entity,val_entry=check_question_answer(val_logs,val_labels)
test_seen_set,test_seen_entity,test_entry=check_question_answer(test_seen_logs,test_seen_labels)
pure_val=val_entry-train_entry
pure_test=test_entry-train_entry
test_unseen_set,test_unseen_entity,test_unseen_entry=check_question_answer(test_unseen_entity_logs,test_unseen_entity_labels)