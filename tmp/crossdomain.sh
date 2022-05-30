

exp=$1

dataset=amazon_all



if [[ $exp -eq 1 ]]
then
CUDA_VISIBLE_DEVICES=0 python src/train.py --dataset $dataset --model DNN --use_vae 0 --vqvae 0  --dropout_rate 0.3  --domain_num 1 --domain_idx 2 --embedding_dim 64
elif [[ exp -eq 2 ]]
then
    # CUDA_VISIBLE_DEVICES=1 python src/train.py --dataset $dataset --model DNN --use_vae 1 --vqvae 1  --dropout_rate 0.3  --domain_num 1 --domain_idx 1 --embedding_dim 64
    CUDA_VISIBLE_DEVICES=1 python src/train.py --dataset ${dataset} --model DNN --use_vae 1 --vqvae 0  --dropout_rate 0.3  --domain_num 1 --domain_idx 2 --embedding_dim 64 --restore ./best_model/amazon_all_DNN_b128_lr0.001_d64_len20_u2u_cl_0.0u2i_cl_0.0i2i_cl_0.0_use_vae_1domain_num_1_idx_1_amazon_all
elif [[ $exp -eq 3 ]]
then
CUDA_VISIBLE_DEVICES=0 python src/train.py --dataset $dataset --model DNN --use_vae 0 --vqvae 0  --dropout_rate 0.3  --domain_num 1 --domain_idx 1 --embedding_dim 64
elif [[ exp -eq 4 ]]
then
    # CUDA_VISIBLE_DEVICES=1 python src/train.py --dataset $dataset --model DNN --use_vae 1 --vqvae 1  --dropout_rate 0.3  --domain_num 1 --domain_idx 2 --embedding_dim 64
    CUDA_VISIBLE_DEVICES=1 python src/train.py --dataset ${dataset} --model DNN --use_vae 1 --vqvae 0  --dropout_rate 0.3  --domain_num 1 --domain_idx 1 --embedding_dim 64 --restore ./best_model/amazon_all_DNN_b128_lr0.001_d64_len20_u2u_cl_0.0u2i_cl_0.0i2i_cl_0.0_use_vae_1domain_num_1_idx_2_amazon_all
else
    echo "Not implemented exp number"
fi












