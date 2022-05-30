dataset=$1
model_type=$2
topn=$3
max_iter=$4
num_interest=$5
neg_num=$6

u2u_cl=1.0
u2i_cl=1.0
i2i_cl=0.1
echo "dataset: "${dataset}", model_type: "${model_type}", u2u_cl: "${u2u_cl}", u2i_cl: "${u2i_cl}", i2i_cl: "${i2i_cl}
mkdir logs
sh run_train_infer_test.sh ${dataset} ${model_type} ${topn} ${u2u_cl} ${u2i_cl} $i2i_cl $max_iter $num_interest $neg_num |& tee logs/${dataset}_${model_type}_${topn}_u2u_${u2u_cl}_u2i_${u2i_cl}_i2i_${i2i_cl}_max_iter_${max_iter}_neg_num${neg_num}

u2u_cl=0.0
u2i_cl=0.0
i2i_cl=0.0
echo "dataset: "${dataset}", model_type: "${model_type}", u2u_cl: "${u2u_cl}", u2i_cl: "${u2i_cl}", i2i_cl: "${i2i_cl}
mkdir logs
sh run_train_infer_test.sh ${dataset} ${model_type} ${topn} ${u2u_cl} ${u2i_cl} $i2i_cl $max_iter $num_interest $neg_num |& tee logs/${dataset}_${model_type}_${topn}_u2u_${u2u_cl}_u2i_${u2i_cl}_i2i_${i2i_cl}_max_iter_${max_iter}_neg_num${neg_num}