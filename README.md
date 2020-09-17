# Syntax Encoding with Application in Authorship Attribution

dataset and code for the paper: **Syntax Encoding with Application in Authorship Attribution**. Richong Zhang, Zhiyuan Hu, Hongyu Guo, Yongyi Mao. EMNLP 2018 [[pdf](https://www.aclweb.org/anthology/D18-1294/)]

you can get the datasets used in the paper at [here](https://drive.google.com/drive/folders/1hlIWVSt0dfy8fz8d4wRzZItl-LCo5BH1?usp=sharing).

## overview

We propose a novel strategy to encode the syntax parse tree of sentence into a learnable distributed representation. The proposed syntax encoding scheme is provably information lossless. In specific, an embedding vector is constructed for each word in the sentence, encoding the path in the syntax tree corresponding to the word. The one-to-one correspondence between these “syntax-embedding” vectors
and the words (hence their embedding vectors) in the sentence makes it easy to integrate such a representation with all word-level NLP models. We empirically show the benefits of the syntax embeddings on the Authorship Attribution domain, where our approach improves upon the prior art and achieves new
performance records on five benchmarking data sets.

## requirement

- python
- pytorch
- tensorflow
- numpy
- nltk
- standford parser


## execute

```
python main.py --dataset <dataset_name> --num_authors <num_of_authors>
```

* the parameter *dataset* will choose from [blogs, CCAT, imdb]

* the parameter *num_authors* will depend on the dataset you choose. when you choose *blogs* or *CCAT*, the parameter *num_authors* wil be 10 or 50, but the max authors num of *imdb* is 62.


This paper propose a syntax feature encoding method which can be used in . It has been accepted by EMNLP2018

## Citation

```
@inproceedings{zhang2018syntax,
  title={Syntax encoding with application in authorship attribution},
  author={Zhang, Richong and Hu, Zhiyuan and Guo, Hongyu and Mao, Yongyi},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={2742--2753},
  year={2018}
}
```



## Licesen

MIT
