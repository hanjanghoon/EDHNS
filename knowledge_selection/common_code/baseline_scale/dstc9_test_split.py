
import json
from tqdm import tqdm
import re
import numpy as np
import random


with open('dstc9_data/test/logs.json', 'r') as f:
    test_logs = json.load(f)

with open('dstc9_data/test/labels.json', 'r') as f:
    test_labels = json.load(f)

with open('dstc9_data/knowledge.json', 'r') as f:
    kb9_seen = json.load(f)
    seen_kb_entity=[]
    for domain in kb9_seen:
        for id in kb9_seen[domain].keys():
            seen_kb_entity.append(id)

with open('dstc9_data/test/knowledge.json', 'r') as f:
    kb9_test = json.load(f)
    id2domain={}
    for domain in kb9_test:
        for id in kb9_test[domain].keys():
            id2domain[id]=domain


#label은 외부지식 사용할때만....

def split_unseen(test_logs,test_labels):
   
    seen_logs,seen_labels,unseen_entity_logs,unseen_entity_labels,unseen_domain_logs,unseen_domain_labels=[],[],[],[],[],[]

    for d_id,label in enumerate(tqdm(test_labels)):
        if label['target']==True:
            entity_id=str(label['knowledge'][0]['entity_id'])
            if entity_id in seen_kb_entity:
                seen_logs.append(test_logs[d_id]) 
                seen_labels.append(label)
            else:
                if id2domain[entity_id]=='attraction':
                    unseen_domain_logs.append(test_logs[d_id]) 
                    unseen_domain_labels.append(label)
                else:
                    unseen_entity_logs.append(test_logs[d_id]) 
                    unseen_entity_labels.append(label)
    print("seen %d"%len(seen_logs))
    print("unseen_entity %d"%len(unseen_entity_logs))
    print("unseen_domain %d"%len(unseen_domain_logs))
    return  seen_logs,seen_labels,unseen_entity_logs,unseen_entity_labels,unseen_domain_logs,unseen_domain_labels


seen_logs,seen_labels,unseen_entity_logs,unseen_entity_labels,unseen_domain_logs,unseen_domain_labels=split_unseen(test_logs,test_labels)

print()



# with open('dstc9_data/test_seen/logs.json', 'w') as f:
#     json.dump(seen_logs, f, indent=4)
# with open('dstc9_data/test_seen/labels.json', 'w') as f:
#     json.dump(seen_labels, f, indent=4)

# with open('dstc9_data/test_unseen_entity/logs.json', 'w') as f:
#     json.dump(unseen_entity_logs, f, indent=4)
# with open('dstc9_data/test_unseen_entity/labels.json', 'w') as f:
#     json.dump(unseen_entity_labels, f, indent=4)

# with open('dstc9_data/test_unseen_domain/logs.json', 'w') as f:
#     json.dump(unseen_domain_logs, f, indent=4)
# with open('dstc9_data/test_unseen_domain/labels.json', 'w') as f:
#     json.dump(unseen_domain_labels, f, indent=4)

#print(dialog_cnt+1)
#print(mixed_domain_cnt)
print("END Program")

print("end")