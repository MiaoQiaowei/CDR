dataset=$1
model_type=$2
topn=$3
u2u_cl=$4
u2i_cl=$5
i2i_cl=$6
max_iter=$7
num_interest=$8
mkdir ./data/${dataset}_data/predict
python ./src/train.py --dataset ${dataset} --model_type ${model_type} --u2u_cl ${u2u_cl} --u2i_cl ${u2i_cl} --i2i_cl ${i2i_cl} --max_iter ${max_iter} --num_interest ${num_interest}
python ./src/train.py -p predict --dataset ${dataset} --model_type ${model_type} --u2u_cl ${u2u_cl} --u2i_cl ${u2i_cl} --i2i_cl ${i2i_cl} --max_iter ${max_iter} --num_interest ${num_interest}
python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_valid_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn $topn