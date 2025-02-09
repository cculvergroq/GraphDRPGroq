#!/bin/bash

# Below are several examples of how to run the data preprocessing script.
# Currently, only the CSA runs are supported (within-study or cross-study).
# Uncomment and run the one you are you interested in.

# ----------------------------------------
# CSA (cross-study analysis) exmple
# ----------------------------------------

# # Within-study
# python graphdrp_preprocess_improve.py \
#     --train_split_file CCLE_split_0_train.txt \
#     --val_split_file CCLE_split_0_val.txt \
#     --test_split_file CCLE_split_0_test.txt \
#     --ml_data_outdir ml_data/CCLE-CCLE/split_0

# Cross-study
python graphdrp_preprocess_improve.py \
    --train_split_file GDSCv1_split_0_train.txt \
    --val_split_file GDSCv1_split_0_val.txt \
    --test_split_file CCLE_all.txt \
    --ml_data_outdir ml_data/GDSCv1-CCLE/split_0


# ----------------------------------------
# LCA (learning curve analysis) exmple
# ----------------------------------------

# # Train with sample size 1000
# python graphdrp_preprocess_improve.py \
#     --train_split_file CCLE_split_4_train_size_1024.txt \
#     --val_split_file CCLE_split_4_val.txt \
#     --test_split_file CCLE_split_4_test.txt \
#     --ml_data_outdir ml_data/CCLE/split_4/size_1000

# # Train with sample size 8000
# python graphdrp_preprocess_improve.py \
#     --train_split_file CCLE_split_4_train_size_8000.txt \
#     --val_split_file CCLE_split_4_val.txt \
#     --test_split_file CCLE_split_4_test.txt \
#     --ml_data_outdir ml_data/CCLE/split_4/size_8000


# ----------------------------------------
# Error analysis exmple
# ----------------------------------------

# # Train with ...
# python graphdrp_preprocess_improve.py \
#     --train_split_file CCLE_split_4_train.txt \
#     --val_split_file CCLE_split_4_val.txt \
#     --test_split_file CCLE_split_4_test.txt \
#     --ml_data_outdir ml_data/CCLE/split_4


# ----------------------------------------
# Robustness
# ----------------------------------------

# # Train with noise level of 2 added to x data
# python graphdrp_preprocess_improve.py \
#     --train_split_file CCLE_split_4_train.txt \
#     --val_split_file CCLE_split_4_val.txt \
#     --test_split_file CCLE_split_4_test.txt \
#     --ml_data_outdir ml_data/CCLE/split_4/x_noise_2

# # Train with noise level of 7 added to x data
# python graphdrp_preprocess_improve.py \
#     --train_split_file CCLE_split_4_train.txt \
#     --val_split_file CCLE_split_4_val.txt \
#     --test_split_file CCLE_split_4_test.txt \
#     --ml_data_outdir ml_data/CCLE/split_4/x_noise_7
