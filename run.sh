#!/bin/sh
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
