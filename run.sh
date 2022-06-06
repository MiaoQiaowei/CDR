dataset=ml_nf
model=DNN
exp_name=$model-$dataset-wo-vqvae

CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --dataset $dataset\
    --data_path data\
    --model $model\
    --dropout 0.3\
    --embedding_dim 64\
    --save_path save\
    --exp_name test_model_acc\
    --domain_index 0 \
    # --vqvae


