# ICLR: Integrated Contrastive Learning for Recommendation

## train
```
python src/train.py --dataset book --model_type ComiRec-SA --u2u_cl 0 --u2i_cl 0 --i2i_cl 0
```
## inference
```
python src/train.py -p predict --dataset book --model_type ComiRec-SA --u2u_cl 0 --u2i_cl 0 --i2i_cl 0
```
## evaluate u2i, u2u, u2ui, i2i
```
python src/knn_store.py --key_file_regex "data/book_data/predict/predict_*_key.npy" --test_user_embs_file data/book_data/predict/predict_valid_user_key.npy --user_click_file data/book_data/user_clicks.json --topn 20
```
