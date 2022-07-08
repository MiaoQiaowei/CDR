dataset=amazon_all
model=DNN
domain_index=1
exp_name=$model-$dataset-$domain_index

CUDA_VISIBLE_DEVICES=1 python3 train.py \
    --dataset $dataset\
    --data_path data\
    --model $model\
    --embedding_dim 64\
    --embedding_num 64\
    --batch_size 64 \
    --save_path gpu1\
    --exp_name $exp_name\
    --domain_index $domain_index \
    --restore_path gpu1/DNN-amazon_all-0/model.ckpt\
    # --vqvae \
    # --ISCS \
    # 