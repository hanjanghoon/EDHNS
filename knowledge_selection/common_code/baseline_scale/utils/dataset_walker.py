import os
import json

class DatasetWalker(object):
    def __init__(self, dataset, dataroot, labels=False, labels_file=None):
        path = os.path.join(os.path.abspath(dataroot))
            
        if dataset not in ['train', 'val', 'test','test_seen','test_seen_toy','test_unseen_entity','test_unseen_domain',"sampling","ranking"]:
            raise ValueError('Wrong dataset name: %s' % (dataset))

        #jang
        if dataset=="sampling" or dataset=="ranking":
            logs_file = os.path.join(path, 'train', 'logs.json')
        
        else:
            logs_file = os.path.join(path, dataset, 'logs.json')
        
        with open(logs_file, 'r') as f:
            self.logs = json.load(f)

        self.labels = None

        if labels is True:
            
            if labels_file is None:
                #jang sampling cnrk
                if dataset=="sampling" or dataset=="ranking":
                    labels_file = os.path.join(path, 'train', 'labels.json')
                else:
                    labels_file = os.path.join(path, dataset, 'labels.json')

            with open(labels_file, 'r') as f:
                self.labels = json.load(f)

    def __iter__(self):
        if self.labels is not None:
            for log, label in zip(self.logs, self.labels):
                yield(log, label)
        else:
            for log in self.logs:
                yield(log, None)

    def __len__(self, ):
        return len(self.logs)
