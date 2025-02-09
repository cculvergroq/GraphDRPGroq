#!/bin/bash --login

# ===========================================================
## Cross-study generalization (CSG) workflow - within-study
# ===========================================================

# --------------------
## Workflow settings
# --------------------

# Parameters of the experiment/run/workflow
# TODO: this should be stored as the experiment metadata that we can go back check
source_data_name="ccle"
target_data_name=$source_data_name
split=5
# epochs=2
epochs=10
y_col_name=AUC

# MAIN_DATADIR is the dir that stores all the data (IMPROVE_DATA_DIR, CANDLE_DATA_DIR, else)
# TODO: The MAIN_DATADIR and the sub-directories below should standardized. How?
MAIN_DATADIR=improve_data_dir

# Sub-directories
ML_DATADIR=$MAIN_DATADIR/ml_data
MODEL_DIR=$MAIN_DATADIR/models
INFER_DIR=$MAIN_DATADIR/infer

OUTDIR=$ML_DATADIR


# -------------
## Preprocess
# -------------
# TODO: If a model needs info about the target dataset (primarily for CSG), this can be provided as target_data_name.
SPLITDIR_NAME=splits
TRAIN_ML_DATADIR=$ML_DATADIR/data."$source_data_name"/split_"$split"_tr
VAL_ML_DATADIR=$ML_DATADIR/data."$source_data_name"/split_"$split"_vl
TEST_ML_DATADIR=$ML_DATADIR/data."$source_data_name"/split_"$split"_te
python frm_preprocess.py \
    --source_data_name $source_data_name \
    --splitdir_name $SPLITDIR_NAME \
    --split_file_name split_"$split"_tr_id \
    --y_col_name $y_col_name \
    --outdir $TRAIN_ML_DATADIR
# python frm_preprocess.py \
#     --source_data_name $source_data_name \
#     --splitdir_name $SPLITDIR_NAME \
#     --split_file_name split_"$split"_vl_id \
#     --y_col_name $y_col_name \
#     --outdir $VAL_ML_DATADIR
# python frm_preprocess.py \
#     --source_data_name $target_data_name \
#     --splitdir_name $SPLITDIR_NAME \
#     --split_file_name split_"$split"_te_id \
#     --y_col_name $y_col_name \
#     --outdir $TEST_ML_DATADIR


# ------
## HPO
# ------
# TODO: Here should be HPO to determine the best HPs


# # --------
# ## Train
# # --------
# # Train using tr samples
# # Early stop using vl samples
# # Save model to dir that encodes the tr and vl info in the dir name
# MODEL_OUTDIR=$MODEL_DIR/"$source_data_name"/split_"$split"/"tr_vl"
# python frm_train.py \
#     --config_file frm_params.txt \
#     --epochs $epochs \
#     --y_col_name $y_col_name \
#     --train_ml_datadir $TRAIN_ML_DATADIR \
#     --val_ml_datadir $VAL_ML_DATADIR \
#     --model_outdir $MODEL_OUTDIR


# # --------
# ## Infer
# # --------
# model_dir=$MODEL_OUTDIR
# infer_outdir=$INFER_DIR/"$source_data_name-$target_data_name"/split_"$split"
# python frm_infer.py \
#     --config_file frm_params.txt \
#     --test_ml_datadir $TEST_ML_DATADIR \
#     --model_dir $model_dir \
#     --infer_outdir $infer_outdir
