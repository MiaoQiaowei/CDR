dataset=ml_nf
model=DNN
domain_index=0
exp_name=new-ISCS_$model-$dataset-$domain_index

# step 1-domain 0
CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --dataset $dataset\
    --data_path data\
    --model $model\
    --dropout 0.3\
    --embedding_dim 64\
    --save_path save_bias_false\
    --exp_name $exp_name\
    --domain_index $domain_index \
    --vqvae \
    --ISCS \
    # --restore_path save_bias_false/new-ISCS_DNN-ml_nf-0/model.ckpt

