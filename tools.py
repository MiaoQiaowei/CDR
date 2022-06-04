import os
import os.path as osp
import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow


def get_data_info(args):

    data_info = {}
    data_info['path']= osp.join(args.data_path, args.dataset)
    data_info['dataset'] = args.dataset

    if args.dataset == 'ml_nf':
        domain = ['ml','nf']
        data_info['domain'] = domain[args.domain_index]
        data_info['item_count'] = 5569+10
        data_info['max_len'] = 20
        data_info['test_iter'] = 500

    elif args.dataset == 'tc_iqi':
        domain = ['tc', 'iqi']
        data_info['domain'] = domain[args.domain_index]
        data_info['item_count'] = 4871+10
        data_info['max_len'] = 20
        data_info['test_iter'] = 500

    elif args.dataset == 'amazon':
        domain = ['amazon_1', 'amazon_2']
        data_info['domain'] = domain[args.domain_index]
        data_info['item_count'] = 365318+100
        data_info['max_len'] = 20
        data_info['test_iter'] = 2000

    elif args.dataset == 'amazon_all':
        domain = ['amazon_1', 'amazon_2']
        data_info['domain'] = domain[args.domain_index]
        data_info['item_count'] = 217587+100
        data_info['max_len'] = 20
        data_info['test_iter'] = 1000

    else:
        raise ValueError(f'DATA is not supported')

    # if args.vqvae:
    #     data_info['batch_size'] = 32
    # else:
    #     data_info['batch_size'] = 128

    return data_info

def get_model(args):
    if args.model == 'DNN':
        from model import DNN
        model = DNN(args)
    else:
        raise ValueError(f'can not find {args.model}')

    return model    

def save(path, sess):
    if not osp.exists(path):
        os.makedirs(path)
    saver = tf.compat.v1.train.Saver()
    saver.save(sess, osp.join(path, 'model.ckpt'))

def restore(path, sess):
    path = osp.join(path)
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    variables = tf.global_variables()

    var_to_restore = []
    for var in variables:
        if var.name.split(':')[0] in var_to_shape_map:
            var_to_restore.append(var)
    
    saver = tf.train.Saver(var_list=var_to_restore)
    saver.restore(sess,path)

def get_DCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)

def get_NDCG(rank_list, pos_items):
    pos_items_metrics = [x for x in pos_items if x in rank_list]
    relevance = np.ones_like(pos_items_metrics, dtype=float)
    it2rel = {it: r for it, r in zip(pos_items_metrics, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)

    idcg = get_DCG(relevance)

    dcg = get_DCG(rank_scores)

    if dcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg

def make_dir(path):
    if not osp.exists(path):
        os.makedirs(path)

def get_trainable_variables(ignore_names=[]):
    variable_names = [v for v in tf.trainable_variables() if not any(name in v.name for name in ignore_names)]
    return variable_names

