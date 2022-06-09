dataset=ml_nf
model=DNN
domain_index=1
exp_name=$model-$dataset-$domain_index

# step 1-domain 0
CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --dataset $dataset\
    --data_path data\
    --model $model\
    --dropout 0.3\
    --embedding_dim 64\
    --save_path save\
    --exp_name $exp_name\
    --domain_index $domain_index \
    # --vqvae \
    # --ISCS \
    # --restore_path save/DNN-ml_nf-vqvae-ISCS-0/model.ckpt\
