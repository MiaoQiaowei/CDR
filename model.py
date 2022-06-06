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
            # self.embedding_table = tf.get_variable("embedding_table", [self.item_count, self.embedding_dim], trainable=True)

            self.embedding_table_bias = tf.Variable(tf.random_normal([self.item_count], mean=0.0, stddev=0.1,dtype= tf.float32), trainable=True, name='embedding_table_bias')

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

            # loss *= tf.cast(tf.gather(mask, 0), tf.float32)
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

    def attention(self, x):
        bs,domain_num,slen,dim = x.get_shape()
        flattened = tf.reshape(x, [-1, dim])
        bs = tf.shape(x)[0]
        
        # get IS qkv
        IS_query = tf.layers.dense(flattened, dim, activation=None, name='IS_q')
        IS_key = tf.layers.dense(flattened, dim, activation=None, name=f'IS_k')
        IS_value = tf.layers.dense(flattened, dim, activation=None, name=f'IS_v')

        # reshape qkv 2 [bs, domain_num, slen, dim]
        IS_query_ = tf.reshape(IS_query, [bs, slen,dim])
        IS_key_ = tf.reshape(IS_key, [bs, slen, dim])
        IS_value_ = tf.reshape(IS_value, [bs, slen,dim])

        # get IS atten
        IS_kq = tf.matmul(IS_key_, tf.transpose(IS_query_,perm=[0,1,3,2])) # [bs, domain_num, slen, slen]
        IS_A = tf.nn.softmax(IS_kq, axis=-1) #only use softmax on the last dim
        Z = tf.matmul(IS_A, IS_value_) # [bs, domain_num, slen, dim]
        Z = tf.reshape(Z, [bs,  slen, dim])

        # get CS qkv
        CS_query = tf.layers.dense(flattened, dim, activation=None, name=f'CS_q')
        CS_key = tf.layers.dense(flattened, dim, activation=None, name=f'CS_k')
        CS_value = tf.layers.dense(flattened, dim, activation=None, name=f'CS_v')

        other_dim = slen
        CS_kq = tf.matmul( CS_key, tf.transpose(CS_query))

        index = [
            [i for t in range(other_dim)] for i in range(self.batch_size)
        ]

        one_hot = tf.one_hot(index, depth=bs, on_value=1.0)
        one_hot = tf.reshape(one_hot, [bs*other_dim, -1])
        mask = one_hot @ tf.transpose(one_hot,[1,0])
        CS_kq_ = (1-mask) * CS_kq - mask* np.inf
        inf = tf.where(mask>0, tf.fill([bs*other_dim, bs*other_dim],-np.inf), tf.fill([bs*other_dim, bs*other_dim],0.0))
        CS_kq_= (1-mask) * CS_kq + inf

        CS_A = tf.nn.softmax(CS_kq_, axis=-1)#only use softmax on the last dim
        
        X = tf.matmul(CS_A, CS_value)
        X = tf.reshape(X,[bs,  slen, dim])
        return Z, X
    
    def IS(self, x):
        bs,slen,dim = x.get_shape()
        flattened = tf.reshape(x, [-1, dim])
        bs = tf.shape(x)[0]
        other_dim=slen
        
        # get IS qkv
        IS_query = tf.layers.dense(flattened, dim, activation=None, name='IS_q')
        IS_key = tf.layers.dense(flattened, dim, activation=None, name=f'IS_k')
        # IS_value = tf.layers.dense(flattened, dim, activation=None, name=f'IS_v')

        index = [
            [i for t in range(other_dim)] for i in range(self.batch_size)
        ]

        one_hot = tf.one_hot(index, depth=bs, on_value=1.0)
        one_hot = tf.reshape(one_hot, [bs*other_dim, -1])
        mask = one_hot @ tf.transpose(one_hot,[1,0])

        # reshape qkv 2 [bs, domain_num, slen, dim]
        # IS_query_ = tf.reshape(IS_query, [bs,domain_num, slen,dim])
        # IS_key_ = tf.reshape(IS_key, [bs, domain_num, slen, dim])
        # IS_value_ = tf.reshape(IS_value, [bs,domain_num, slen,dim])

        # get IS atten
        IS_kq = tf.matmul(IS_key, tf.transpose(IS_query)) # [bs*domain_num*slen, bs*domain_num*slen]
        IS_kq = mask*IS_kq - (1-mask)*np.inf
        IS_A = tf.nn.softmax(IS_kq, axis=-1) #only use softmax on the last dim

        Z = tf.matmul(IS_A, flattened) # [bs, domain_num, slen, dim]
        Z = tf.reshape(Z, [bs,  slen, dim])
        return Z
    
    def CS(self, x):
        bs,domain_num,slen,dim = x.get_shape()
        flattened = tf.reshape(x, [-1, dim])
        bs = tf.shape(x)[0]

        # get CS qkv
        CS_query = tf.layers.dense(flattened, dim, activation=None, name=f'CS_q')
        CS_key = tf.layers.dense(flattened, dim, activation=None, name=f'CS_k')
        # CS_value = tf.layers.dense(flattened, dim, activation=None, name=f'CS_v')

        other_dim = slen
        CS_kq = tf.matmul( CS_key, tf.transpose(CS_query))

        index = [
            [i for t in range(other_dim)] for i in range(self.batch_size)
        ]

        one_hot = tf.one_hot(index, depth=bs, on_value=1.0)
        one_hot = tf.reshape(one_hot, [bs*other_dim, -1])
        mask = one_hot @ tf.transpose(one_hot,[1,0])
        CS_kq_ = (1-mask) * CS_kq - mask* np.inf
        # inf = tf.where(mask>0, tf.fill([bs*other_dim, bs*other_dim],-np.inf), tf.fill([bs*other_dim, bs*other_dim],0.0))
        # CS_kq_= (1-mask) * CS_kq + inf

        CS_A = tf.nn.softmax(CS_kq_, axis=-1)#only use softmax on the last dim
        
        X = tf.matmul(CS_A, flattened)
        X = tf.reshape(X,[bs,  slen, dim])
        return X   
    
    def restore(self, path, sess):
        saver = saver = tf.train.Saver()
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
