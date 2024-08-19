import json
from tqdm import tqdm
import re
import numpy as np
import random

with open('dstc9_data/train/logs.json', 'r') as f:
    train_logs = json.load(f)

with open('dstc9_data/train/labels.json', 'r') as f:
    train_labels = json.load(f)

refine_logs=[]
refine_labels=[]
for log,label in zip(train_logs,train_labels):
    if label['target']==True:
        refine_logs.append(log)
        refine_labels.append(label)
with open('dstc9_data/train/logs.json', 'w') as f:
    json.dump(refine_logs, f, indent=4)
with open('dstc9_data/train/labels.json', 'w') as f:
    json.dump(refine_labels, f, indent=4)