import numpy as np
from tqdm import tqdm
import pickle


def make_data(infile,outfile):
    with open(infile, 'rb') as f:
        data=pickle.load(f)
   
    # lines=f.readlines()
    multi_choice_data=[]
    pos_list=[]
    neg_list=[]
    for i, triple in enumerate(tqdm(data)):
        label=triple[0]
        context=triple[1:-1]
        response=triple[-1]
        if label=='1':
            pos_resp=response
        else:
            neg_list.append(response)

        if i%10==9:
            multi_choice_data.append([context,pos_resp,neg_list])
            neg_list=[]

    print("making dialog complete!!!!!!!!!")
    with open(outfile, 'wb') as f:
        pickle.dump(multi_choice_data, f)
    return data



dialog_dict=make_data("ubuntu_data/valid.pkl","ubuntu_data/valid_mdns.pkl")
dialog_dict=make_data("ubuntu_data/test.pkl","ubuntu_data/test_mdns.pkl")
# count_average("data/train_query_pos_neg.pkl")
print()