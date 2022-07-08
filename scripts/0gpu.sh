dataset=ml_nf
model=DNN
domain_index=1
exp_name=$model-$dataset-$domain_index-item

CUDA_VISIBLE_DEVICES=1 python3 train.py \
    --dataset $dataset\
    --data_path data\
    --model $model\
    --embedding_dim 64\
    --embedding_num 64\
    --save_path gpu0\
    --exp_name $exp_name\
    --domain_index $domain_index \
    --meta_item \
    --restore_path gpu0/DNN-ml_nf-0-item/model.ckpt\
    # --meta_user \
    # --restore_path gpu0/DNN-ml_nf-0-user/model.ckpt\
    # --vqvae \
    # --ISCS \
    