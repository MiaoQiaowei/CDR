dataset=book1
model_type=ComiRec-SA
topn=50
u2u_cl=1.0
u2i_cl=1.0
i2i_cl=0.2
max_iter=100
num_interest=4
neg_num=30

rm -rf ./data/${dataset}_data/predict
rm -rf ./data/${dataset}_data/mic_negs.json
rm -rf ./data/${dataset}_data/user_clicks.json
rm -rf ./data/${dataset}_data/predict/*

mkdir ./data/${dataset}_data/predict
#coef=0.1
python ./src/train.py -p predict --dataset ${dataset} --model_type ${model_type} --u2u_cl ${u2u_cl} --u2i_cl ${u2i_cl} --i2i_cl ${i2i_cl} --max_iter ${max_iter} --num_interest ${num_interest} --dropout_rate 0.3
#weight=0.0
#python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight ${weight} --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef ${coef}
#weight=0.5
#python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight ${weight} --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef ${coef}
#weight=1.0
#python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight ${weight} --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef ${coef}
#weight=2.0
#python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight ${weight} --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef ${coef}
#weight=3.0
#python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight ${weight} --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef ${coef}
#weight=4.0
#python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight ${weight} --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef ${coef}
#weight=5.0
#python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight ${weight} --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef ${coef}
#weight=10.0
#python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight ${weight} --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef ${coef}
#
#coef=0.0
#weight=2.0
#python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight ${weight} --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef ${coef}
#coef=0.05
#python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight ${weight} --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef ${coef}
#coef=0.1
#python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight ${weight} --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef ${coef}
#coef=0.15
#python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight ${weight} --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef ${coef}
#coef=0.2
#python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight ${weight} --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef ${coef}
#coef=0.25
#python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight ${weight} --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef ${coef}
#coef=0.3
#python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight ${weight} --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef ${coef}
#coef=0.35
#python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight ${weight} --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef ${coef}
#coef=0.4
#python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight ${weight} --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef ${coef}

weight=2.0
coef=0.1
#python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight ${weight} --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef ${coef} --use_u2i 0
#python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight ${weight} --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef ${coef} --use_u2u 0
python ./src/knn_store.py --key_file_regex "./data/"${dataset}"_data/predict/predict_*_key.npy" --test_user_embs_file "./data/"${dataset}"_data/predict/predict_test_user_key.npy" --user_click_file "./data/"${dataset}"_data/user_clicks.json" --topn 50 --weight ${weight} --cate_file ./data/${dataset}_data/${dataset}_item_cate.txt --coef ${coef} --use_i2i 0