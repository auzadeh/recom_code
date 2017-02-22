#!/bin/bash

for d in {1..10}; do
    for c in {3..10}; do
        for place in nv az; do
            python train_word2vec.py \
                raw_data/${place}_train_seq.tsv \
                rest_embedding/varying_d_and_c/${place}_train_${d}d_${c}c.dat ${d} ${c}

            python run_experiment.py raw_data/${place}_train_user_rest_rating_time.tsv \
                rest_embedding/varying_d_and_c/${place}_train_${d}d_${c}c.dat \
                raw_data/${place}_test_user_rest_rating_time.tsv \
                performance/${place}_errors_${d}d_${c}c.tsv
        done
    done
done
