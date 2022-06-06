dataset=ml_nf
model=DNN
domain_index=0
exp_name=test
# step 1-domain 0
CUDA_VISIBLE_DEVICES=1 python3 train.py \
    --dataset $dataset\
    --data_path data\
    --model $model\
    --dropout 0.3\
    --embedding_dim 64\
    --save_path save\
    --exp_name $exp_name\
    --domain_index $domain_index \
    --restore_path save/DNN-ml_nf-wo-vqvae-0-init_func/model.ckpt\
    # --lower_boundary -3.059065103530884 \
    # --upper_boundary 2.7775886058807373 \
    # --stddev 0.8039396405220032
