conda activate mqw
CUDA_VISIBLE_DEVICES=0 python src/train.py --dataset ml_nf --model DNN --use_vae 1 --vqvae 1  --dropout_rate 0.3  --domain_num 1 --domain_idx 2 --embedding_dim 64