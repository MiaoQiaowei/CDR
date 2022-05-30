dataset=ml_nf
CUDA_VISIBLE_DEVICES=1 python src/train.py --dataset ml_nf --model DNN --use_vae 1 --vqvae 1  --dropout_rate 0.3  --domain_num 1 --domain_idx 2 --embedding_dim 64 -use_ISCS 0 -exp_name $dataset-single_domain
#  2>&1 | tee logs/out.log