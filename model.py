from turtle import update
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
        self.args = args
        self.loss = None
        self.opt = None
        self.step = 0
        self.merge = None

    def see(self):
        grads = tf.gradients(self.loss, tf.trainable_variables())

        for grad, var in zip(grads, tf.trainable_variables()):
            if grad is None:
                raise ValueError(f'{var.name} has no gradient')
            tf.summary.histogram(var.name + '/gradient', grad)

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
            # self.fixed_len_target_ids = tf.placeholder(tf.int32, [None, args.domain_num, None], name='seq_targets')

            self.history_item_ids = tf.placeholder(tf.int32, [None, self.max_len], name='history_item_ids')
            self.history_item_masks = tf.placeholder(tf.int32, [None, self.max_len], name='history_item_masks')

            self.lr = tf.placeholder(tf.float64, [])
            self.dropout_rate = tf.placeholder(tf.float32, [])
            self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        

        # embedding table
        with tf.name_scope('embedding_table'):
            self.embedding_table = tf.Variable(tf.random_normal([self.item_count, self.embedding_dim], mean=0.0, stddev=0.01,dtype= tf.float32), trainable=True, name='embedding_table')
            self.embedding_table_bias = tf.Variable(tf.random_normal([self.item_count], mean=0.0, stddev=0.01,dtype= tf.float32), trainable=True, name='embedding_table_bias')

        # code book
        with tf.name_scope('code_book'):
            self.code_book = tf.get_variable("code_book", [self.embedding_num, self.embedding_dim], trainable=True,
                                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None, dtype=tf.dtypes.float32))

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

            # tf.summary.histogram('weight', self.weight)
            # tf.summary.histogram('bias', self.bias)

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

        loss, _, summary = sess.run(
            [self.loss, self.opt, self.merge],
            feed_dict=feed_dict
        )

        return loss, summary
    
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

class DNN(Model):
    def __init__(self, args, name='DNN'):
        super(DNN,self).__init__(args)
        self.loss = self.create_forward_path(args, name)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
    
    def create_forward_path(self, args, name='DNN'):
        with tf.variable_scope(name, tf.AUTO_REUSE):
            mask = tf.cast(tf.greater_equal(self.history_item_masks, 1), tf.float32)
            
            # get embedding
            # item_embeddings = tf.nn.embedding_lookup(self.embedding_table, self.item_ids)
            
            histroy_item_embeddings = tf.nn.embedding_lookup(self.embedding_table, self.history_item_ids)

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
            # tf.summary.histogram('histroy_item_embeddings_mean',self.histroy_item_embeddings_mean)

            # get loss
            loss = self.sampled_softmax_loss(self.histroy_item_embeddings_mean, self.item_ids)

            tf.summary.scalar('loss', loss)

            self.merge = tf.summary.merge_all()
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

class DNN3():
    def __init__(self, args, name='DNN'):
        self.args = args
        self.loss = None
        self.opt = None
        self.step = 0
        self.merge = None
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
            # self.fixed_len_target_ids = tf.placeholder(tf.int32, [None, args.domain_num, None], name='seq_targets')

            self.history_item_ids = tf.placeholder(tf.int32, [None, self.max_len], name='history_item_ids')
            self.history_item_masks = tf.placeholder(tf.int32, [None, self.max_len], name='history_item_masks')

            self.lr = tf.placeholder(tf.float64, [])
            self.dropout_rate = tf.placeholder(tf.float32, [])
            self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        

        # embedding table
        with tf.name_scope('embedding_table'):
            # self.embedding_table = tf.Variable(tf.random_normal([self.item_count, self.embedding_dim], mean=0.0, stddev=0.01,dtype= tf.float32), trainable=True, name='embedding_table')
            ones = np.ones([self.item_count, self.embedding_dim])
            self.embedding_table = tf.get_variable("embedding_table", [self.item_count, self.embedding_dim],      trainable=True,initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=1, dtype=tf.dtypes.float32))
            self.embedding_table_bias = tf.get_variable("embedding_table_bias", [self.item_count],      trainable=True,initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=1, dtype=tf.dtypes.float32))
            # tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None, dtype=tf.dtypes.float32)

        # code book
        # with tf.name_scope('code_book'):
        #     self.code_book = tf.get_variable("code_book", [self.embedding_num, self.embedding_dim], trainable=True,
        #                                     initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None, dtype=tf.dtypes.float32))

        with tf.variable_scope('DNN', tf.AUTO_REUSE):
            mask = tf.cast(tf.greater_equal(self.history_item_masks, 1), tf.float32)
            
            # get embedding
            # item_embeddings = tf.nn.embedding_lookup(self.embedding_table, self.item_ids)
            
            histroy_item_embeddings = tf.nn.embedding_lookup(self.embedding_table, self.history_item_ids)

            # histroy_item_embeddings = tf.nn.dropout(histroy_item_embeddings ,rate=args.dropout)
            histroy_item_embeddings *= tf.reshape(mask, (-1, self.max_len, 1))

            mask = tf.reshape(mask, [-1,  self.max_len])
            histroy_item_embeddings = tf.reshape(histroy_item_embeddings, [-1, self.max_len, self.embedding_dim])
            masks = tf.concat([tf.expand_dims(mask, -1) for _ in range(self.embedding_dim)], axis=-1)
            histroy_item_embeddings_mean = tf.reduce_sum(histroy_item_embeddings, 1) / (tf.reduce_sum(tf.cast(masks, dtype=tf.float32), 1) + 1e-9)


            # print(histroy_item_embeddings_mean.shape)
            # exit()
            histroy_item_embeddings_mean = tf.Variable(tf.random_normal([self.args.batch_size, self.embedding_dim], mean=0.0, stddev=0.01,dtype= tf.float32), trainable=True, name='histroy_item_embeddings_mean_test')

            # use vqvae
            if args.vqvae:
                vqvae_item_embeddings_mean = self.vqvae(histroy_item_embeddings, self.code_book, mask)
                histroy_item_embeddings_mean = tf.concat([vqvae_item_embeddings_mean, histroy_item_embeddings_mean], axis=-1)

            # get logtis
            with tf.name_scope('encode_layers'):
                histroy_item_embeddings_mean = tf.layers.dense(histroy_item_embeddings_mean, 
                        self.embedding_dim, 
                        activation=None, 
                        name='encoder_layer_0'
                    )
                histroy_item_embeddings_mean = tf.reshape(histroy_item_embeddings_mean, [-1, self.embedding_dim])
            self.histroy_item_embeddings_mean = histroy_item_embeddings_mean
            # tf.summary.histogram('histroy_item_embeddings_mean',self.histroy_item_embeddings_mean)

        # self.weight = tf.get_variable("loss_weight", [self.item_count, self.embedding_dim], trainable=1,
        #                                     initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None, dtype=tf.dtypes.float32))

        # self.bias = tf.get_variable("loss_bias", [self.item_count], trainable=True,
        #                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None, dtype=tf.dtypes.float32))

        # tf.summary.histogram('weight', self.weight)
        # tf.summary.histogram('bias', self.bias)

        x = self.histroy_item_embeddings_mean
        y = self.item_ids
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
        self.loss = tf.reduce_mean(loss)

            # get loss
        # self.loss = self.sampled_softmax_loss(self.histroy_item_embeddings_mean, self.item_ids)

        tf.summary.scalar('loss', self.loss)

        grads = tf.gradients(self.loss, tf.trainable_variables())
        # Summarize all gradients
        for v in tf.trainable_variables():
            print(v.name)
        # exit()
        for grad, var in zip(grads, tf.trainable_variables()):
            # print(type(grad))
            # exit()
            if grad is None:
                print('not find')
                print(var.name + '/gradient')
                # exit()
                continue
            tf.summary.histogram(var.name + '/gradient', grad)
        # print(grads)
        # exit()
            
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.grads_and_vars = self.opt.compute_gradients(loss)

        self.train = self.opt.apply_gradients(self.grads_and_vars)
            
        # self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     optimizer = tf.train.MomentumOptimizer(0.01, momentum=0.01, use_nesterov= True, name='Momentum') #定义优化器
            
        #     grads_and_vars = optimizer.compute_gradients(self.loss) #计算梯度
    
        #     update_op = optimizer.apply_gradients(grads_and_vars) #定义反向传播操作

        #     grads_and_vars = self.opt.compute_gradients(self.loss)
        #     for g, v in grads_and_vars:
        #             print("*****", v.name, g) 

        self.merge = tf.summary.merge_all()
    
    # def create_forward_path(self, args, name='DNN'):
        
    
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
        # print(inputs[4])
        # print(np.array(inputs[4]).shape)
        # exit()

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

        loss, _, summary = sess.run(
            [self.loss, self.train, self.merge],
            feed_dict=feed_dict
        )


        return loss, summary
