from ast import arg
import os
from this import d
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import json
import numpy as np
import random
import sys
import shutil
import tensorflow as tf
import time
import os.path as osp

from data_iterator import DataIterator
from manager import *
from tensorboardX import SummaryWriter
from tqdm import tqdm
from tools import get_data_info, get_model


def train(
    train_path,
    val_path,
    test_path,
    args,
    model,
    manager
):
    manager.logger.info("max_iter: {}".format(args.max_iter))
    save_path = f'{args.save_path}/{args.exp_name}/'

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
            data_path=train_path,
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

        iter_counter = 0
        loss_sum = 0.
        loss_dict = {}
        drop_times = 0

        for X, Y in tqdm(train_loader):
            user_ids, item_ids, domain_ids, fixed_len_item_ids = X
            history_items, history_mask = Y
            loss = model.train(sess, [])




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('-dataset', type=str)
    parser.add_argument('-data_path', type=str)
    parser.add_argument('-save_path', type=str)
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
    args = parser.parse_args()

    
    # set helper
    begin_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    args.exp_name = f'{args.exp_name}_{begin_time}'
    manager = Manager(f'{args.save_path}/{args.exp_name}/run.log')

    trainset_name = get_data_info(args.dataset, args.domain_index)
    info_dict = {}
    info_dict['trainset']  = trainset_name
    info_dict.update(vars(args))

    manager.logger.info(f'trainset:{trainset_name}')
    manager.logger.info(sys.argv)
    manager.logger.info(info_dict)
    
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

    train(train_path,val_path,test_path, model, manager)




