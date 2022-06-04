dataset=ml_nf
rm -rf save
CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --dataset $dataset\
    --data_path data\
    --model DNN\
    --dropout 0.3\
    --embedding_dim 64\
    --save_path save\
    --exp_name test_model_acc\
    --domain_index 0 \
    # --vqvae

# dataset=ml_nf
# exp='XZ_loss_all_test'
# #step 1
# CUDA_VISIBLE_DEVICES=1 python src/train.py \
#     --dataset $dataset \
#     --model DNN --use_vae 0 --vqvae 0  --dropout_rate 0.3  \
#     --domain_num 1 --domain_idx 1 --embedding_dim 64 -use_ISCS 0 \
#     -exp_name $dataset-$exp-step1
