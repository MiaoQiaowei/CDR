import argparse
import pprint 
import faiss
import numpy as np
import random
import sys
import tensorflow as tf
import time
import os.path as osp

from data_iterator import DataIterator
from manager import Manager
from tqdm import tqdm
from tools import get_NDCG, get_data_info, get_model, restore, save

tf.logging.set_verbosity(tf.logging.INFO)

def eval(loader, model, sess, manager:Manager, args, name='val'):
    # manager.logger.info(f'eval on domain:{args.domain_index}')

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
        f'{name}_num':0.,
        f'{name}_recall':0.0,
        f'{name}_ndcg':0.0,
        f'{name}_hit_rate':0.0,
    }

    for X, Y in tqdm(loader):
        user_ids, item_ids, domain_ids = X
        history_items, history_mask = Y

        if len(history_items) == 0:
            continue
        
        history_embeddings = model.get_history_embeddings(sess, [history_items, history_mask, domain_ids, len(user_ids)])[0]

        history_embeddings = np.array(history_embeddings)

        # dist, index
        D, I = gpu_index.search(history_embeddings, args.topN) 
        for index, item_ids_pre_user in enumerate(item_ids):
            rank = I[index].tolist()
            metric[f'{name}_ndcg'] += get_NDCG(rank, item_ids_pre_user)
            
            recall_num = len(set(rank) & set(item_ids_pre_user))
            metric[f'{name}_recall'] += recall_num/len(item_ids_pre_user)

            metric[f'{name}_hit_rate'] += 1 if recall_num > 0 else 0

        metric[f'{name}_num'] += len(item_ids)

    num = metric[f'{name}_num']

    for k in metric.keys():
        if k != f'{name}_num':
            metric[k] /= num
    
    return metric

def train(train_loader, val_loader, model, sess, manager:Manager, args):
    patience = 0

    for X, Y in tqdm(train_loader):
        user_ids, item_ids, domain_ids = X
        history_items, history_mask = Y

        inputs = [
            user_ids, item_ids, domain_ids,
            history_items, history_mask,
            args.lr, args.dropout, len(user_ids)
        ]

        loss = model.run(sess, inputs)

        manager.add(loss)

        if manager.global_step % args.test_iter== 0:

            manager.logger.info(f'step:{manager.global_step}')

            metric = eval(val_loader, model, sess, manager, args, name='val')
            
            metric['train_loss'] = manager.avg()
            metric['ce_loss'] = 0
            metric['vqvae_loss'] = 0
            manager.info['lowerboundary'] = float(model.lowerboundary)
            manager.info['upperboundary'] = float(model.upperboundary)
            manager.info['stddev'] = float(model.stddev)

            manager.logger.info(', '.join([f'{key}: %.6f' % value for key, value in metric.items() if 'num' not in key]))
            manager.logger.info(f'lowerboundary:{model.lowerboundary}  upperboundary:{model.upperboundary} stddev:{model.stddev}')
            
            for k,v in metric.items():
                manager.writer.add_scalar(tag=k, scalar_value=v,global_step=manager.global_step / args.test_iter)
            manager.info.update(metric)
            
            if metric['val_recall'] > manager.info['best_recall']:
                save(args.save_path, sess)
                manager.info['best_recall'] = metric['val_recall']
            
            if manager.info['best_recall'] <= metric['val_recall']:
                patience = 0
            else:
                patience += 1
                if patience > args.patience:
                    break

            manager.clean()
        
        if manager.counter >= args.max_step:
            break

        

def test(loader, model, sess, manager:Manager, args):
    manager.logger.info(f'testing!')

    restore_path = osp.join(args.save_path, 'model.ckpt')
    restore(restore_path, sess)

    metric = eval(loader, model, sess, manager, args, name='test')
    manager.logger.info(', '.join([f'{key}: %.6f' % value for key, value in metric.items()]))

    manager.info.update(metric)
    manager.record(manager.info)

def get_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataset', type=str, default='ml_nf')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--save_path', type=str, default='save')
    parser.add_argument('--restore_path', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--domain_num', type=int, default=1)
    parser.add_argument('--domain_index', type=int, default=0)

    # model
    parser.add_argument('--model', type=str,default='DNN')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--embedding_num', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--ISCS', action='store_true', default=False)
    parser.add_argument('--vqvae', action='store_true', default=False)
    # parser.add_argument('--self_attn', action='store_true', default=False)
    parser.add_argument('--upper_boundary', type=float, default=1)
    parser.add_argument('--lower_boundary', type=float, default=-1)
    parser.add_argument('--stddev', type=float, default=0.1)

    # run
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--only_test_last_one', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--coef', type=float,default=0.0)
    parser.add_argument('--topN', type=int, default=50)
    parser.add_argument('--max_step', type=int, default=300000)
    args = parser.parse_args()

    return args

def main(_):
    args = get_args()

    # set helper
    args.begin_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))
    args.save_path = osp.join(args.save_path, args.exp_name)
    manager = Manager(args.save_path)
    
    # set seed 
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # get data
    data_info = get_data_info(args)
    train_path = osp.join(data_info['path'], 'train.json')
    val_path = osp.join(data_info['path'], 'valid.json')
    test_path = osp.join(data_info['path'], 'test.json')

    args.item_count = data_info['item_count']
    args.max_len = data_info['max_len']
    args.test_iter = data_info['test_iter']

    # get model
    model = get_model(args)

    manager.logger.info("max_iter: {}".format(args.max_step))
    manager.info.update(vars(args))
    all_info = pprint.pformat(manager.info)
    manager.logger.info(all_info)

    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # from tensorflow.python import debug as tf_debug

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

        if args.restore_path != '':
            manager.logger.info(f'restore model from {args.restore_path}')
            # restore(args.restore_path, sess)
            restore(args.restore_path, sess, ignore=['embedding'])

        train(train_loader, val_loader, model, sess, manager, args)

        test(test_loader, model, sess, manager, args)

if __name__ == '__main__':
    tf.app.run()

    
    

