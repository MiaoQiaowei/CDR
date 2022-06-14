import tensorflow as tf
import numpy as np
import os.path as osp
import os

def get_cos_similarity(x1, x2, dim_num=4, eye=True):
        #[num, dim]
        if dim_num == 4:
            (num , domain_num, slen, dim) = x1.get_shape().as_list()
        else:
            (num,dim) = x1.get_shape().as_list()
        x1 = tf.reshape(x1, [-1, dim])
        x2 = tf.reshape(x2, [-1, dim])
        # num = x1.get_shape()[0]
        num = 128

        X1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=1)) #[num ,1]
        X2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2), axis=1))
        # 内积
        X1_X2 = tf.matmul(x1, tf.transpose(x1)) #[num, num]
        X1_X2_norm = tf.matmul(tf.reshape(X1_norm,[-1,1]),tf.reshape(X2_norm,[1,-1]))
        # 计算余弦距离
        cos = X1_X2/X1_X2_norm

        if eye is True:
            # diag = tf.linalg.band_part(cos,0,0)
            if dim_num == 4:
            # (num , domain_num, slen, dim) = x1.get_shape().as_list()
                other_dim = num*domain_num*slen
            else:
                # (num,dim) = x1.get_shape().as_list()
                other_dim = num
            mask = tf.eye(num_rows=other_dim, num_columns=other_dim)
            cos = tf.reduce_sum(tf.abs(mask*cos))/tf.reduce_sum(mask)
            # cos = tf.reduce_sum(mask*cos)/tf.reduce_sum(mask)
        else:
            diag = tf.linalg.band_part(cos,0,-1)
            item_nums = num*(num-1)/2
            cos = (tf.reduce_sum(tf.abs(diag))-num)/item_nums
            # cos = (tf.reduce_sum(diag)-num)/item_nums
            cos = tf.reduce_mean(cos)
        return cos

class Model:
    def __init__(self, args):
        self.create_model_variable(args)
        self.args = args
        self.loss = None
        self.opt = None
        self.step = 0

    def create_model_variable(self, args):
        self.item_count = args.item_count
        self.embedding_dim = args.embedding_dim
        self.embedding_num = args.embedding_num 
        self.domain_num = args.domain_num
        self.max_len = args.max_len

        # placeholder
        with tf.name_scope('inputs'):
            self.user_ids = tf.placeholder(tf.int32, [None,], name='user_ids')
            self.item_ids = tf.placeholder(tf.int32, [None,], name='item_ids')
            self.domain_labels = tf.placeholder(tf.int32, [None,], name='domain_labels')

            self.history_item_ids = tf.placeholder(tf.int32, [None, self.max_len], name='history_item_ids')
            self.history_item_masks = tf.placeholder(tf.int32, [None, self.max_len], name='history_item_masks')

            self.lr = tf.placeholder(tf.float64, [])
            self.dropout_rate = tf.placeholder(tf.float32, [])
            self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')

        # code book
        with tf.name_scope('code_book_vars'):
            self.code_book = tf.get_variable("code_book", [self.embedding_num, self.embedding_dim], trainable=True,
                                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None, dtype=tf.dtypes.float32))

        # embedding table
        with tf.name_scope('embedding_table_vars'):
            upper = args.upper_boundary
            lower = args.lower_boundary
            mean = (upper+lower)/2
            stddev = args.stddev

            self.embedding_table = tf.Variable(tf.random_normal([self.item_count, self.embedding_dim], mean=mean, stddev=stddev,dtype= tf.float32), trainable=True, name='embedding_table')

            self.embedding_table_bias = tf.Variable(tf.zeros([self.item_count], dtype=tf.float32), trainable=False, name='embedding_table_bias')

    def create_forward_path(self, args):
        raise NotImplementedError

    def encoder(self, x):
        with tf.name_scope('encode_layers'):
            x = tf.layers.dense(x, 
                    self.embedding_dim, 
                    activation=None, 
                    name='encoder_layer_0'
                )
            x = tf.reshape(x, [-1, self.embedding_dim])
        return x

    def vqvae(self, x, code_book, mask=None):
        # vqvae
        with tf.name_scope('vqvae'):
            
            if self.args.ISCS:
                # vq_x, encodings = self.get_quantized(x, code_book)
                CS = self.CS(x)
                IS = self.IS(x, code_book)
                # IS = self.CS(x, mask=1)

                CS_ = tf.reshape(CS, [-1, self.embedding_dim])
                IS_ = tf.reshape(IS, [-1, self.embedding_dim])
                # ISCS = tf.concat([IS_, CS_], axis=-1)
                # vq_x = tf.layers.dense(ISCS, self.embedding_dim, activation=None, name='mix_CS_and_IS')
                # vq_x = tf.contrib.layers.layer_norm(vq_x)
                
                ISCS = tf.concat([IS_, CS_], axis=-1)
                vq_x = tf.layers.dense(ISCS, self.embedding_dim, activation=None, name='mix_CS_and_IS')
                vq_x = tf.contrib.layers.layer_norm(vq_x)
                # vq_x, encodings = self.get_quantized(x, code_book)
            else:
                vq_x, encodings = self.get_quantized(x, code_book)
            
            vq_x = tf.reshape(vq_x, [-1, self.max_len, self.embedding_dim])
            vq_mean = tf.reduce_sum(vq_x, 1) / tf.reshape(tf.reduce_sum(mask, axis=-1), [-1, 1])
            
        return vq_mean
    
    def sampled_softmax_loss(self, x, y, mask=None):
        with tf.name_scope('sample_softmax_loss'):

            y = tf.reshape(y, [-1, 1])

            loss = tf.nn.sampled_softmax_loss(
                weights=self.embedding_table,
                biases=self.embedding_table_bias,
                inputs=x,
                labels=y,
                num_sampled=30 * self.args.batch_size,
                num_classes=self.item_count
            )

            loss = tf.reduce_mean(loss)
 
        return loss  

    def run(self, sess, inputs):
        
        self.step += 1

        feed_dict = {
            self.user_ids:inputs[0],
            self.item_ids:inputs[1],
            self.domain_labels:inputs[2],

            self.history_item_ids:inputs[3],
            self.history_item_masks:inputs[4],

            self.lr:inputs[5],
            self.dropout_rate:inputs[6],
            self.batch_size:inputs[7]
        }

        loss, _ = sess.run(
            [self.loss, self.opt],
            feed_dict=feed_dict
        )

        return loss
    
    def get_quantized(self, x, code_book):
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])
        distances = (
                tf.reduce_sum(flattened ** 2, axis=1, keepdims=True)
                + tf.reduce_sum(code_book ** 2, axis=1)
                - 2 * tf.matmul(flattened, code_book, transpose_b=True)
        )
        encoding_indices = tf.argmin(distances, axis=1)
        encodings = tf.one_hot(encoding_indices, self.embedding_num)
        quantized = tf.matmul(encodings, code_book)
        quantized = tf.reshape(quantized, input_shape)
        return quantized, encodings

    def qkv(self, q, k, v, dim):
        with tf.variable_scope('qkv',reuse=tf.AUTO_REUSE):
            q = tf.layers.dense(q, dim, activation=None, name='q')
            k = tf.layers.dense(k, dim, activation=None, name='k')
            v = tf.layers.dense(v, dim, activation=None, name='v')
            return q, k, v

    def CS(self, x, mask=0):
        '''
        估算每个x在train set 中的比例（在batch 中）
        '''

        slen = self.args.max_len
        dim = self.args.embedding_dim

        with tf.name_scope('CS'):
            flattened = tf.reshape(x, [-1, dim])

            q, k, v = self.qkv(flattened, flattened, flattened, dim)

            qk = q @ tf.transpose(k)/(self.embedding_dim ** 0.5) # bs*slen , bs*slen

            mask = tf.range(slen)
            mask = tf.expand_dims(mask,axis=1)
            mask = tf.tile(mask,[1,self.batch_size])
            mask = tf.one_hot(mask, depth=self.batch_size, on_value=1.0)
            mask = tf.reshape(mask, [self.batch_size*slen, -1])
            mask = mask @ tf.transpose(mask,[1,0])
            inf_mask = tf.zeros_like(mask) + float("-inf")
            if mask == 0:
                bool_mask = tf.cast(1-mask, tf.bool)
            else:
                bool_mask = tf.cast(mask, tf.bool)
            qk = tf.where(bool_mask, x=qk, y=inf_mask)

            A = tf.nn.softmax(qk, axis=-1) # bs*slen, bs*slen
            CS = A@v

            return tf.reshape(CS, [-1, slen, dim])
    
    def IS(self, x, code_book):
        '''
        估算返回 x通过code_book找到的中介变量
        '''
        bs = self.args.batch_size
        slen = self.args.max_len
        dim = self.args.embedding_dim

        with tf.name_scope('IS'):
            flattened = tf.reshape(x, [-1, dim])

            q, k, v = self.qkv(flattened, code_book, code_book, dim)

            qk = q @ tf.transpose(k)/(self.embedding_dim ** 0.5)  # bs*slen , embedding_num
            IS = qk @ v # bs*slen , dim

            return tf.reshape(IS, [-1, slen, dim])

    def restore(self, path, sess):
        saver = tf.train.Saver()
        saver.restore(sess, path)
    
    def save(self, path, sess):
        if not osp.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, osp.join(path, 'model.ckpt'))


class DNN(Model):
    def __init__(self, args, name='DNN'):
        super(DNN,self).__init__(args)
        self.lowerboundary = args.lower_boundary
        self.upperboundary = args.upper_boundary
        self.stddev = 0
        self.loss = self.create_forward_path(args, name)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        
    def create_forward_path(self, args, name='DNN'):
        with tf.variable_scope(name, tf.AUTO_REUSE):
            mask = tf.cast(tf.greater_equal(self.history_item_masks, 1), tf.float32)
            
            # get embedding
            histroy_item_embeddings = tf.nn.embedding_lookup(self.embedding_table, self.history_item_ids)

            self.upperboundary_tf = tf.reduce_max(histroy_item_embeddings)
            self.lowerboundary_tf = tf.reduce_min(histroy_item_embeddings)
            self.stddev_tf = tf.math.reduce_std(histroy_item_embeddings)

            # histroy_item_embeddings = tf.nn.dropout(histroy_item_embeddings ,rate=args.dropout)
            histroy_item_embeddings *= tf.reshape(mask, (-1, self.max_len, 1))

            mask = tf.reshape(mask, [-1,  self.max_len])
            histroy_item_embeddings = tf.reshape(histroy_item_embeddings, [-1, self.max_len, self.embedding_dim])
            masks = tf.concat([tf.expand_dims(mask, -1) for _ in range(self.embedding_dim)], axis=-1)
            histroy_item_embeddings_mean = tf.reduce_sum(histroy_item_embeddings, 1) / (tf.reduce_sum(tf.cast(masks, dtype=tf.float32), 1) + 1e-9)

            # use vqvae
            if args.vqvae:
                vqvae_item_embeddings_mean = self.vqvae(histroy_item_embeddings, self.code_book, mask)
                histroy_item_embeddings_mean = tf.concat([vqvae_item_embeddings_mean, histroy_item_embeddings_mean], axis=-1)

            # get logtis
            self.histroy_item_embeddings_mean = self.encoder(histroy_item_embeddings_mean)

            # get loss
            loss = self.sampled_softmax_loss(self.histroy_item_embeddings_mean, self.item_ids)

            return loss
    
    def get_history_embeddings(self, sess, inputs):
        feed_dict = {
            self.domain_labels:inputs[2],

            self.history_item_ids:inputs[0],
            self.history_item_masks:inputs[1],

            self.batch_size:inputs[3]
        }
        history_embeddings = sess.run(
            [self.histroy_item_embeddings_mean],
            feed_dict=feed_dict
        )
        return history_embeddings
        
    
    def run(self, sess, inputs):
    
        self.step += 1

        feed_dict = {
            self.user_ids:inputs[0],
            self.item_ids:inputs[1],
            self.domain_labels:inputs[2],

            self.history_item_ids:inputs[3],
            self.history_item_masks:inputs[4],

            self.lr:inputs[5],
            self.dropout_rate:inputs[6],
            self.batch_size:inputs[7]
        }

        loss, _ , upperboundary, lowerboundary, stddev= sess.run(
            [self.loss, self.opt, self.upperboundary_tf, self.lowerboundary_tf, self.stddev_tf],
            feed_dict=feed_dict
        )

        self.upperboundary = max(upperboundary, self.upperboundary)
        self.lowerboundary = min(lowerboundary, self.lowerboundary)
        self.stddev = max(stddev, self.stddev)

        return loss
