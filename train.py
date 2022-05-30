import argparse
import faiss
import numpy as np
import random
import sys
import tensorflow as tf
import time
import os.path as osp

from data_iterator import DataIterator
from manager import *
from tqdm import tqdm
from tools import get_NDCG, get_data_info, get_model, save, restore

tf.logging.set_verbosity(tf.logging.INFO)

def eval(loader, model, sess, manager:Manager, args, name='val'):
    manager.logger.info(f'eval on domain:{args.domain_index}')

    embedding_table = sess.run(model.embedding_table)

    try:
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        try:
            gpu_index = faiss.GpuIndexFlatIP(res, args.embedding_dim, flat_config)
            gpu_index.add(embedding_table)
        except Exception as e:
            manager.logger.info("gpu error for faiss search: {}".format(e))
            return {}
    except:
        gpu_index = faiss.IndexFlatIP(args.embedding_dim)
        gpu_index.add(embedding_table)

    metric = {
        f'{name}_num':0,
        f'{name}_recall':0.0,
        f'{name}_ndcg':0.0,
        f'{name}_hit_rate':0.0,
    }

    for X, Y in tqdm(loader):
        user_ids, item_ids, domain_ids = X
        history_items, history_mask = Y

        if len(history_items) == 0:
            continue
        
        history_embeddings = model.get_history_embeddings(sess, [history_embeddings, history_mask, domain_ids])

        if len(history_embeddings.shape) == 3:
            history_embeddings = history_embeddings[:, 1, :].copy(order='C')
        else:
            history_embeddings = history_embeddings[:, 1, :, :].copy(order='C')

        # dist, index
        D, I = gpu_index.search(history_embeddings, args.topN) 
        for index, item_ids_pre_user in enumerate(item_ids):
            rank = I[index].tolist()
            metric['ndcg'] += get_NDCG(rank, item_ids_pre_user)
            
            recall_num = len(set(rank) & set(item_ids_pre_user))
            metric['recall'] += recall_num/len(item_ids_pre_user)

            metric['hit_rate'] += 0 if recall_num > 0 else 1

        metric['num'] += len(item_ids)
    
    num = metric['num']
    for k in metric.keys():
        if k != 'num':
            metric[k] /= num
    
    return metric

def train(train_loader, val_loader, model, sess, manager:Manager, args):
    for X, Y in tqdm(train_loader):
        user_ids, item_ids, domain_ids = X
        history_items, history_mask = Y
        loss = model.run(
            sess,
            [
                user_ids, item_ids, domain_ids,
                history_items, history_mask,
                args.lr, args.dropout, args.batch_size
            ]
        )

        manager.add(loss)

        if manager.counter % args.val_frequency:
            manager.logger(f'loss:{manager.avg()}')

            metric = eval(val_loader, model, sess, manager, args, name='val')

            manager.logger.info(', '.join([f'{key}: %.6f' % value for key, value in metric.items()]))

            metric['train_loss'] = manager.avg()

            for k,v in metric.items():
                manager.writer.add_scalar(k, v, manager.counter)
        
        if manager.counter >= args.max_step:
            break
            
        

def test(loader, model, sess, manager:Manager, args):
    manager.logger.info(f'testing!')

    restore(args.save_path,sess)

    metric = eval(loader, model, sess, manager, args, name='test')
    manager.logger.info(', '.join([f'{key}: %.6f' % value for key, value in metric.items()]))

    manager.info.update(metric)
    manager.record(manager.info)

def get_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('-dataset', type=str)
    parser.add_argument('-data_path', type=str)
    parser.add_argument('-save_path', type=str)
    parser.add_argument('--restore_path', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--domain_num', type=int, default=2)
    parser.add_argument('--domain_index', type=int, default=0)

    # model
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--hidden_layers', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--restore',  type=str, default="")
    parser.add_argument('--ISCS', action='store_true', default=False)
    parser.add_argument('--vqvae', action='store_true', default=False)
    parser.add_argument('--vae', action='store_true', default=False)
    parser.add_argument('--u2u_cl',  type=float, default=0.0)
    parser.add_argument('--u2i_cl', type=float, default=0.0)
    parser.add_argument('--i2i_cl', type=float, default=0.0)

    # run
    parser.add_argument('-exp_name', type=str)
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--only_test_last_one', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--restore', type=str,default="")
    parser.add_argument('--coef', type=float,default=0.0)
    parser.add_argument('--topN', type=int, default=50)
    parser.add_argument('--val_frequency', type=int, default=50)
    parser.add_argument('--max_step', type=int, default=100000)
    args = parser.parse_args()

    return args

def main():
    args = get_args()

    # set helper
    begin_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    args.exp_name = f'{args.exp_name}_{begin_time}'
    manager = Manager(f'{args.save_path}/{args.exp_name}/run.log')

    trainset_name = get_data_info(args.dataset, args.domain_index)
    manager.info['trainset']  = trainset_name
    manager.info.update(vars(args))

    manager.logger.info(f'trainset:{trainset_name}')
    manager.logger.info(sys.argv)
    # manager.logger.info(info_dict)
    
    # set seed 
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # get data
    data_info = get_data_info(args)
    train_path = osp.join(data_info['path'], 'train.json')
    val_path = osp.join(data_info['path'], 'valid.json')
    test_path = osp.join(data_info['path'], 'test.json')

    # get model
    model = get_model(args)

    manager.logger.info("max_iter: {}".format(args.max_iter))
    args.save_path = f'{args.save_path}/{args.exp_name}'

    gpu_options = tf.GPUOptions(allow_growth=True)
    # item_cate_map = load_item_cate(cate_file)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_loader = DataIterator(
            data_path=train_path,
            batch_size=args.batch_size,
            max_len=args.max_len,
            domain_num=args.domain_num,
            domain_index=args.domain_index,
            is_train=True,
            use_vqvae=args.vqvae
        )

        val_loader = DataIterator(
            data_path=val_path,
            batch_size=args.batch_size,
            max_len=args.max_len,
            domain_num=args.domain_num,
            domain_index=args.domain_index,
            is_train=False,
            use_vqvae=args.vqvae
        )

        test_loader = DataIterator(
            data_path=test_path,
            batch_size=args.batch_size,
            max_len=args.max_len,
            domain_num=args.domain_num,
            domain_index=args.domain_index,
            is_train=False,
            use_vqvae=args.vqvae
        )

        # init
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        manager.logger.info('train begin')
        sys.stdout.flush()

        if args.restore != '':
            manager.logger.info(f'restore model from {args.restore}')
            restore(args.restore, sess)

        train(train_loader, model, sess, manager, args)

        test(test_loader, model, sess, manager, args)


if __name__ == '__main__':
    tf.app.run()

    
    




