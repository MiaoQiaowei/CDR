dataset=ml_nf
model=DNN
domain_index=0
exp_name=$model-$dataset-wo-vqvae-$domain_index

CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --dataset $dataset\
    --data_path data\
    --model $model\
    --dropout 0.3\
    --embedding_dim 64\
    --save_path save\
    --exp_name $exp_name\
    --domain_index $domain_index \
    # --vqvae


