import tensorflow as tf
import numpy as np

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
        # self.loss, self.opt = self.create_forward_path(args)
        self.loss = None
        self.opt = None

    def create_model_variable(self, args):
        self.item_count = args.item_count
        self.embedding_dim = args.embedding_dim
        self.embedding_num = 512 
        self.domain_num = args.domain_num
        self.max_len = args.max_len

        # placeholder
        with tf.name_scope('inputs'):
            self.user_ids = tf.placeholder(tf.int32, [None, ], name='user_ids')
            self.item_ids = tf.placeholder(tf.int32, [None, args.domain_num], name='item_ids')
            self.domain_labels = tf.placeholder(tf.int32, [None, args.domain_num], name='domain_labels')
            # self.fixed_len_target_ids = tf.placeholder(tf.int32, [None, args.domain_num, None], name='seq_targets')

            self.history_item_ids = tf.placeholder(tf.int32, [None, args.domain_num, self.max_len], name='history_item_ids')
            self.history_item_masks = tf.placeholder(tf.int32, [None, args.domain_num, self.max_len], name='history_item_masks')

            self.lr = tf.placeholder(tf.float64, [])
            self.dropout_rate = tf.placeholder(tf.float32, [])
            self.batch_size = tf.placeholder(tf.int32, [args.batch_size], name='batch_size')

        # embedding table
        self.embedding_tabel = tf.Variable([args.item_count, self.embedding_dim],trainable=True,name='embedding_table')

        # code book
        self.code_book = tf.get_variable("code_book", [self.embedding_num, self.embedding_dim], trainable=True,
                                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None, dtype=tf.dtypes.float32))

    def create_forward_path(self, args):
        # mask = tf.cast(tf.greater_equal(self.domain_num, 1), tf.float32)
        # mask_len = tf.cast(tf.reduce_sum(self.mask, -1), dtype=tf.int32)
        
        # # get embedding
        # item_embeddings = tf.nn.embedding_lookup(self.embedding_tabel, self.item_ids)
        # target_item_embeddings = tf.nn.embedding_lookup(self.embedding_tabel, self.target_item_ids)

        # # use vqvae
        # if args.vqvae:
        #     item_embeddings = self.vqvae(item_embeddings, self.code_book)

        # # get logtis
        # logtis = self.encoder(item_embeddings)

        # # get loss
        # loss = self.sample_softmax_loss(logtis, self.item_ids, mask=mask)

        # # get opt
        # opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        # return loss, opt
        raise NotImplementedError

    def encoder(self, x):
        x = tf.layers.dense(x, 
                self.domain_num*self.embedding_num, 
                activation=None, 
                name='encoder_layer_0'
            )
        x = tf.reshape(x, [-1, self.domain_num, self.embedding_dim])
        return x

    def vqvae(self, x, code_book):
        # vqvae
        with tf.name_scope('vqvae'):
            vq_x, encodings = self.get_quantized(x, code_book)
            vq_x = tf.reshape(vq_x, [-1, self.domain_num*self.max_len, self.embedding_dim])
            vq_mean = tf.reduce_sum(vq_x, 1) / tf.reshape(tf.reduce_sum(self.mask, axis=-1), [-1, 1])
            vq_mean = tf.concat([vq_mean, self.item_his_eb_mean], axis=-1)
        return vq_mean
    
    def sample_softmax_loss(self, x, y, mask):
        weight = tf.Variable(
            [self.embedding_num, self.embedding_dim],
            trainable=True,
            name='weight'
        )

        bias = tf.Variable(
            [self.embedding_num],
            initial_value=tf.zeros_initializer(),
            trainable=True
        )

        loss = tf.nn.sampled_softmax_loss(
            weight=weight,
            bias=bias,
            labels=y,
            inputs=x
        )

        loss *= tf.cast(tf.gather(mask, 0), tf.float32)
        loss = tf.reduce_mean(loss)
        return loss  

    def run(self, sess, inputs):

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
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
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
        IS_query_ = tf.reshape(IS_query, [bs,domain_num, slen,dim])
        IS_key_ = tf.reshape(IS_key, [bs, domain_num, slen, dim])
        IS_value_ = tf.reshape(IS_value, [bs,domain_num, slen,dim])

        # get IS atten
        IS_kq = tf.matmul(IS_key_, tf.transpose(IS_query_,perm=[0,1,3,2])) # [bs, domain_num, slen, slen]
        IS_A = tf.nn.softmax(IS_kq, axis=-1) #only use softmax on the last dim
        Z = tf.matmul(IS_A, IS_value_) # [bs, domain_num, slen, dim]
        Z = tf.reshape(Z, [bs, domain_num, slen, dim])

        # get CS qkv
        CS_query = tf.layers.dense(flattened, dim, activation=None, name=f'CS_q')
        CS_key = tf.layers.dense(flattened, dim, activation=None, name=f'CS_k')
        CS_value = tf.layers.dense(flattened, dim, activation=None, name=f'CS_v')

        other_dim = domain_num*slen
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
        X = tf.reshape(X,[bs, domain_num, slen, dim])
        return Z, X
    
    def IS(self, x):
        bs,domain_num,slen,dim = x.get_shape()
        flattened = tf.reshape(x, [-1, dim])
        bs = tf.shape(x)[0]
        other_dim=domain_num*slen
        
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
        Z = tf.reshape(Z, [bs, domain_num, slen, dim])
        return Z
    
    def CS(self, x):
        bs,domain_num,slen,dim = x.get_shape()
        flattened = tf.reshape(x, [-1, dim])
        bs = tf.shape(x)[0]

        # get CS qkv
        CS_query = tf.layers.dense(flattened, dim, activation=None, name=f'CS_q')
        CS_key = tf.layers.dense(flattened, dim, activation=None, name=f'CS_k')
        # CS_value = tf.layers.dense(flattened, dim, activation=None, name=f'CS_v')

        other_dim = domain_num*slen
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
        X = tf.reshape(X,[bs, domain_num, slen, dim])
        return X
    

class DNN(Model):
    def __init__(self, args, name='DNN'):
        super(DNN).__init__(args)
        self.loss = self.create_forward_path(args, name)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
    
    def create_forward_path(self, args, name='DNN'):
        with tf.variable_scope(name, tf.AUTO_REUSE):
            mask = tf.cast(tf.greater_equal(self.domain_num, 1), tf.float32)
            
            # get embedding
            item_embeddings = tf.nn.embedding_lookup(self.embedding_tabel, self.item_ids)
            histroy_item_embeddings = tf.nn.embedding_lookup(self.embedding_tabel, self.history_item_ids)
            histroy_item_embeddings = tf.nn.dropout(histroy_item_embeddings ,rate=args.dropout)
            histroy_item_embeddings *= tf.reshape(mask, (-1, self.domain_num, self.max_len, 1))

            mask = tf.reshape(self.mask, [-1, self.domain_num * self.max_len])
            histroy_item_embeddings = tf.reshape(histroy_item_embeddings, [-1, self.domain_num * self.max_len, self.embedding_dim])
            masks = tf.concat([tf.expand_dims(mask, -1) for _ in range(self.embedding_dim)], axis=-1)
            histroy_item_embeddings = tf.reduce_sum(histroy_item_embeddings, 1) / (tf.reduce_sum(tf.cast(masks, dtype=tf.float32), 1) + 1e-9)

            # use vqvae
            if args.vqvae:
                histroy_item_embeddings = self.vqvae(histroy_item_embeddings, self.code_book)

            # get logtis
            self.histroy_item_embeddings = self.encoder(histroy_item_embeddings)

            # get loss
            loss = self.sample_softmax_loss(self.histroy_item_embeddings, item_embeddings, mask=mask)

            return loss
    
    def get_history_embeddings(self, sess, feed_dict):
        history_embeddings = sess.run(
            [self.histroy_item_embeddings],
            feed_dict=feed_dict
        )
        return history_embeddings

