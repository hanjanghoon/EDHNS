Efficient Dynamic Hard Negative Sampling for Dialogue Selection <img src="https://pytorch.org/assets/images/logo-dark.svg" width = "90" align=center />
====================================
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fine-grained-post-training-for-improving/conversational-response-selection-on-ubuntu-1)](https://paperswithcode.com/sota/conversational-response-selection-on-ubuntu-1?p=fine-grained-post-training-for-improving)


Implements the model described in the following paper [Efficient Dynamic Hard Negative Sampling for Dialogue Selection](https://aclanthology.org/2024.nlp4convai-1.6/) in ACL-NLP4ConvAI 2024.

```
@inproceedings{han-etal-2024-efficient,
    title = "Efficient Dynamic Hard Negative Sampling for Dialogue Selection",
    author = "Han, Janghoon  and Lee, Dongkyu  and Shin, Joongbo  and Bae, Hyunkyung  and Bang, Jeesoo  and Kim, Seonghwan and Choi, Stanley Jungkyu  and Lee, Honglak",
    booktitle = "Proceedings of the 6th Workshop on NLP for Conversational AI (NLP4ConvAI 2024)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.nlp4convai-1.6",
    pages = "89--100",
}
```
![fig2](https://github.com/user-attachments/assets/6d5274f3-51ba-4ce6-a8ed-285947d3edb1)

Setup and Dependencies
----------------------

This code is implemented using PyTorch v1.10.0, and provides out of the box support with CUDA 11.3
Anaconda is the recommended to set up this codebase.
```
# https://pytorch.org
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```


Preparing Data and Checkpoints
-------------

### Dataset and Checkpoints

#### Dataset
- [Dataset for Knowledge Selection (DSTC9, DSTC10)][1]
- [Dataset for Response Selection (Ubuntu, E-commerce)][2]

Original version for each dataset is availble in [Ubuntu Corpus V1][3], [E-Commerce Corpus][4], respectively.

#### Checkpoints
- [Checkpoints (RoBERTa-large-EHDNS) for Knowledge Selection (DSTC9, DSTC10)][5]
- [Checkpoints (BERT-FP-EHDNS) for Response Selection (Ubuntu, E-commerce)][6]


Training
--------

### Preprocess Data
#### For Knowledge Selection
```
DSTC9, DSTC10 dataset include processing python files.
```
#### For Response Selection
```
response_selection/ubuntu/preprocess_FT_ecom.py
response_selection/e-commerce/preprocess_FT_ecom.py
```


### Traing and Test
#### Training (DSTC9, DSTC10, Ubuntu Corpus V1, E-commerce Corpus)
```shell
sh knowledge_selection/dstc9/train_dstc9_rlm_EDHNS.sh
sh knowledge_selection/dstc10/train_dstc10_rlm_EDHNS.sh
sh response_selection/ubuntu/train_bert_ubuntu.sh
sh response_selection/e-commerce/train_bert_ecom.sh
```

#### Test (DSTC9, DSTC10, Ubuntu Corpus V1, E-commerce Corpus)
```shell
sh knowledge_selection/dstc9/test_dstc9_rlm_EDHNS.sh
sh knowledge_selection/dstc10/test_dstc10_rlm_EDHNS.sh
sh response_selection/ubuntu/test_bert_ubuntu.sh
sh response_selection/e-commerce/test_bert_ecom.sh
```

Performance
----------
#### For Knowledge Selection

|DSTC9           | R@1   | R@5   | MRR@5   |
| -------------- | ----- | ----- | ----- |
| [RoBERTa-large-EDHNS] | 0.931 | 0.998 | 0.962 |

| DSTC10         | R@1   | R@2   | R@5   |
| -------------- | ----- | ----- | ----- |
| [RoBERTa-large-EDHNS] | 0.821 | 0.935 | 0.869 |


#### For Response Selection
| Ubuntu         | R@1   | R@2   | R@5   |
| -------------- | ----- | ----- | ----- |
| [BERT_FP-EDHNS] | 0.917 | 0.965 | 0.994 |

| E-Commerce     | R@1   | R@2   | R@5   |
| -------------- | ----- | ----- | ----- |
| [BERT_FP-EDHNS] | 0.957 | 0.986 | 0.997 |


[1]: https://github.com/taesunwhang/BERT-ResSel
[2]: https://drive.google.com/file/d/1-4E0eEjyp7n_F75TEh7OKrpYPK4GLNoE/view?usp=sharing
[3]: https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip
[4]: https://github.com/cooelf/DeepUtteranceAggregation
[5]: https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip
[6]: https://github.com/cooelf/DeepUtteranceAggregation
