dataset=ml_nf
exp='XZ_loss_all_test'
#step 1
CUDA_VISIBLE_DEVICES=1 python src/train.py \
    --dataset $dataset \
    --model DNN --use_vae 0 --vqvae 0  --dropout_rate 0.3  \
    --domain_num 1 --domain_idx 1 --embedding_dim 64 -use_ISCS 0 \
    -exp_name $dataset-$exp-step1

# # step 2
# CUDA_VISIBLE_DEVICES=1 python3 src/train.py\
#     --dataset $dataset\
#     --model DNN --use_vae 1 --vqvae 1  --dropout_rate 0.3\
#     --domain_num 1 --domain_idx 2 --embedding_dim 64\
#     --restore best_model/ml_nf_DNN_b128_lr0.001_d64_len20_u2u_cl_0.0u2i_cl_0.0i2i_cl_0.0_use_vae_1_use_ISCS_1_domain_num_1_idx_1_ml_nf\
#     -use_ISCS 1 -exp_name $dataset-$exp-step2
