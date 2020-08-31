
# A Label Attention Model for ICD Coding from Clinical Text

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
- nltk==3.4.1
- psycopg2==2.7.7
- gensim==3.6.0

Run `pip install -r requirements.txt` to install the required libraries

- Run `python3` and run `import nltk` and `nltk.download('punkt')` for tokenization 

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


**Disclaim:** _The results reported in our paper are averaged over ten (10) randomly initialised runs. Re-run the code might lead to _**slightly**_ different results._
