dataset=book
model_type=ComiRec-SA
topn=50
u2u_cl=1.0
u2i_cl=1.0
i2i_cl=0.1
max_iter=100
num_interest=4
rm -rf ./data/${dataset}_data/predict
mkdir ./data/${dataset}_data/predict
python ./src/train.py --dataset ${dataset} --model_type ${model_type} --u2u_cl ${u2u_cl} --u2i_cl ${u2i_cl} --i2i_cl ${i2i_cl} --max_iter ${max_iter} --num_interest ${num_interest} --dropout_rate 0.3
python ./src/train.py -p predict --dataset ${dataset} --model_type ${model_type} --u2u_cl ${u2u_cl} --u2i_cl ${u2i_cl} --i2i_cl ${i2i_cl} --max_iter ${max_iter} --num_interest ${num_interest} --dropout_rate 0.3
python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight 2.0 --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef 0.01
u2u_cl=0.0
u2i_cl=1.0
i2i_cl=0.1
max_iter=100
num_interest=4
rm -rf ./data/${dataset}_data/predict
mkdir ./data/${dataset}_data/predict
python ./src/train.py --dataset ${dataset} --model_type ${model_type} --u2u_cl ${u2u_cl} --u2i_cl ${u2i_cl} --i2i_cl ${i2i_cl} --max_iter ${max_iter} --num_interest ${num_interest} --dropout_rate 0.3
python ./src/train.py -p predict --dataset ${dataset} --model_type ${model_type} --u2u_cl ${u2u_cl} --u2i_cl ${u2i_cl} --i2i_cl ${i2i_cl} --max_iter ${max_iter} --num_interest ${num_interest} --dropout_rate 0.3
python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight 2.0 --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef 0.01
u2u_cl=1.0
u2i_cl=0.0
i2i_cl=0.1
max_iter=100
num_interest=4
rm -rf ./data/${dataset}_data/predict
mkdir ./data/${dataset}_data/predict
python ./src/train.py --dataset ${dataset} --model_type ${model_type} --u2u_cl ${u2u_cl} --u2i_cl ${u2i_cl} --i2i_cl ${i2i_cl} --max_iter ${max_iter} --num_interest ${num_interest} --dropout_rate 0.3
python ./src/train.py -p predict --dataset ${dataset} --model_type ${model_type} --u2u_cl ${u2u_cl} --u2i_cl ${u2i_cl} --i2i_cl ${i2i_cl} --max_iter ${max_iter} --num_interest ${num_interest} --dropout_rate 0.3
python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight 2.0 --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef 0.01
u2u_cl=1.0
u2i_cl=1.0
i2i_cl=0.0
max_iter=100
num_interest=4
rm -rf ./data/${dataset}_data/predict
mkdir ./data/${dataset}_data/predict
python ./src/train.py --dataset ${dataset} --model_type ${model_type} --u2u_cl ${u2u_cl} --u2i_cl ${u2i_cl} --i2i_cl ${i2i_cl} --max_iter ${max_iter} --num_interest ${num_interest} --dropout_rate 0.3
python ./src/train.py -p predict --dataset ${dataset} --model_type ${model_type} --u2u_cl ${u2u_cl} --u2i_cl ${u2i_cl} --i2i_cl ${i2i_cl} --max_iter ${max_iter} --num_interest ${num_interest} --dropout_rate 0.3
python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight 2.0 --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef 0.01
u2u_cl=1.0
u2i_cl=1.0
i2i_cl=0.1
max_iter=100
num_interest=4
rm -rf ./data/${dataset}_data/predict
mkdir ./data/${dataset}_data/predict
python ./src/train.py --dataset ${dataset} --model_type ${model_type} --u2u_cl ${u2u_cl} --u2i_cl ${u2i_cl} --i2i_cl ${i2i_cl} --max_iter ${max_iter} --num_interest ${num_interest} --dropout_rate 0.0
python ./src/train.py -p predict --dataset ${dataset} --model_type ${model_type} --u2u_cl ${u2u_cl} --u2i_cl ${u2i_cl} --i2i_cl ${i2i_cl} --max_iter ${max_iter} --num_interest ${num_interest} --dropout_rate 0.0
python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight 2.0 --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef 0.01