
# A Label Attention Model for ICD Coding from Clinical Text <a href="https://twitter.com/intent/tweet?text=LAAT%20%28A%20Label%20Attention%20Model%20for%20ICD%20Coding%20from%20Clinical%20Text%29%20Code:&url=https%3A%2F%2Fgithub.com%2Faehrc%2FLAAT"><img alt="Twitter" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2F"></a>  
  
<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/aehrc/LAAT"> <a href="https://github.com/aehrc/LAAT/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/aehrc/LAAT"></a> <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/aehrc/LAAT"> <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/aehrc/LAAT"> <a href="https://github.com/aehrc/LAAT/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/aehrc/LAAT"></a> <a href="https://github.com/aehrc/LAAT/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/aehrc/LAAT"></a>    

This project provides the code for our JICAI 2020 [A Label Attention Model for ICD Coding from Clinical Text](https://arxiv.org/abs/2007.06351) paper.

The general architecture and experimental results can be found in our [paper](https://arxiv.org/abs/2007.06351):

```
  @inproceedings{ijcai2020-461-vu,
      title     = {A Label Attention Model for ICD Coding from Clinical Text},
      author    = {Vu, Thanh and Nguyen, Dat Quoc and Nguyen, Anthony},
      booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, {IJCAI-20}},             
      pages     = {3335--3341},
      year      = {2020},
      month     = {7},
      note      = {Main track}
      doi       = {10.24963/ijcai.2020/461},
      url       = {https://doi.org/10.24963/ijcai.2020/461},
   }
```

**Please CITE** our paper when this code is used to produce published results or incorporated into other software.

### Requirements

- python>=3.6
- pytorch==1.4.0
- scikit-learn==0.23.1
- numpy==1.16.3
- scipy==1.2.1
- pandas==0.24.2
- tqdm==4.31.1
- nltk>=3.4.5
- psycopg2==2.7.7
- gensim==3.6.0

Run `pip install -r requirements.txt` to install the required libraries

Run `python3` and run `import nltk` and `nltk.download('punkt')` for tokenization 

### Data preparation

#### MIMIC-III-full and MIMIC-III-50 experiments
`data/mimicdata/mimic3`
 
- The id files are from [caml-mimic](https://github.com/jamesmullenbach/caml-mimic)
- Install the MIMIC-III database with PostgreSQL following this [instruction](https://mimic.physionet.org/tutorials/install-mimic-locally-ubuntu/)
- Generate the train/valid/test sets using `src/util/mimiciii_data_processing.py`. (Configure the connection to PostgreSQL at Line 139)

#### MIMIC-II-full experiment
`data/mimicdata/mimic2`

- Place the MIMIC-II file ([MIMIC_RAW_DSUMS](https://archive.physionet.org/works/ICD9CodingofDischargeSummaries/)) to `data/mimicdata/mimic2`
- Generate the train/valid/test sets using `src/util/mimicii_data_processing.py`.

**Note that:** The code will generate 3 files (`train.csv`, `valid.csv`, and `test.csv`) for each experiment.

### Pretrained word embeddings 
`data/embeddings`

We used `gensim` to train the embeddings (`word2vec` model) using the entire MIMIC-III discharge summary data. 

Our code also supports subword embeddings (`fastText`) which helps produce better performances (see `src/args_parser.py`).

### How to run

The problem and associated configurations are defined in `configuration/config.json`. Note that there are 3 files in each data folder (`train.csv`, `valid.csv` and `test.csv`)

There are common hyperparameters for all the models and the model-specific hyperparameters. See `src/args_parser.py` for more detail

Here is an example of using the framework on MIMIC-III dataset (full codes) with hierarchical join learning

```
python -m src.run \
    --problem_name mimic-iii_2_full \
    --max_seq_length 4000 \
    --n_epoch 50 \
    --patience 5 \
    --batch_size 8 \
    --optimiser adamw \
    --lr 0.001 \
    --dropout 0.3 \
    --level_projection_size 128 \
    --main_metric micro_f1 \
    --embedding_mode word2vec \
    --embedding_file data/embeddings/word2vec_sg0_100.model \
    --attention_mode label \
    --d_a 512 \
    RNN  \
    --rnn_model LSTM \
    --n_layers 1 \
    --bidirectional 1 \
    --hidden_size 512 
```

