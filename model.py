import tensorflow as tf
import os.path as osp
import os

class Model:
    def __init__(self, args):
        self.create_model_variable(args)
        self.args = args
        self.loss = None
        self.opt = None
        self.step = 0

    def create_model_variable(self, args):
        self.item_count = args.item_count
        self.user_count = args.user_count
        self.embedding_dim = args.embedding_dim
        self.embedding_num = args.embedding_num
        self.code_book_dim = self.embedding_num * 8
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
            self.item_book = tf.get_variable("item_book", [self.embedding_num, self.code_book_dim], trainable=True,
                                            initializer=tf.random_normal_initializer(mean=0.0, stddev=1, seed=None, dtype=tf.float32))
            self.user_book = tf.get_variable("user_book", [self.embedding_num, self.code_book_dim], trainable=True,
                                            initializer=tf.random_normal_initializer(mean=0.0, stddev=1, seed=None, dtype=tf.float32))

        # embedding table
        with tf.name_scope('embedding_table_vars'):
            upper = args.upper_boundary
            lower = args.lower_boundary
            mean = (upper+lower)/2
            stddev = args.stddev

            self.item_embedding_table = tf.Variable(tf.random_normal([self.item_count, self.embedding_dim], mean=mean, stddev=stddev,dtype= tf.float32), trainable=True, name='item_embedding_table')

            self.embedding_table_bias = tf.Variable(tf.zeros([self.item_count], dtype=tf.float32), trainable=False, name='embedding_table_bias')

    def create_forward_path(self, args):
        raise NotImplementedError

    def mixer(self, x, out_dim, name=''):
        with tf.name_scope(f'mix_layers_{name}'):
            x = tf.layers.dense(x, 
                    out_dim, 
                    activation=None, 
                    name=f'mix_layer_{name}_0'
                )
            x = tf.reshape(x, [-1, out_dim])
        return x
    
    def encoder(self, x, fc_or_conv='conv'):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE) as encoder_var:
            if fc_or_conv == 'fc':
                x = tf.reshape(x, [-1, self.embedding_dim])
                x = tf.layers.dense(x, self.embedding_dim//2, activation=None)
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x)
                x = tf.layers.dense(x, self.embedding_dim//4, activation=None)
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x)
                x = tf.layers.dense(x, self.code_book_dim, activation=None)
            else:
                H = int(self.max_len ** 0.5)
                W = self.max_len // H
                x = tf.reshape(x, [-1, H, W, self.embedding_dim])

                x = tf.layers.conv2d(x, filters=self.embedding_dim * 2, kernel_size=2)
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x)
                x = tf.layers.conv2d(x, filters=self.embedding_dim * 4, kernel_size=2)
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x)
                x = tf.layers.conv2d(x, filters=self.embedding_dim * 8, kernel_size=2)

        return x, encoder_var
    
    def decoder(self, x, fc_or_conv='conv'):
        with tf.variable_scope('decoder') as decoder_var:
            if fc_or_conv == 'fc':
                x = tf.layers.dense(x, self.embedding_dim//4, activation=None)
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x)
                x = tf.layers.dense(x, self.embedding_dim//2, activation=None)
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x)
                x = tf.layers.dense(x, self.embedding_dim, activation=None)
            else:
                x = tf.layers.conv2d_transpose(x, filters=self.embedding_dim * 4, kernel_size=2)
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x)
                x = tf.layers.conv2d_transpose(x, filters=self.embedding_dim * 2, kernel_size=2)
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x)
                x = tf.layers.conv2d_transpose(x, filters=self.embedding_dim, kernel_size=2)

        return x, decoder_var

    def vqvae(self, x, code_book, mask=None):

        with tf.name_scope('vqvae'):
            x_shape = tf.shape(x)
            x_ = tf.reshape(x, [-1, self.embedding_dim])
            
            x_encode,var_encoder = self.encoder(x)
            x_decode, encodings = self.get_quantized(x_encode, code_book)
            vq_x_,var_decoder = self.decoder(x_encode + tf.stop_gradient(x_decode-x_encode))

            vq_x = tf.reshape(vq_x_, x_shape)
            vq_mean = tf.reduce_sum(vq_x, 1) / tf.reshape(tf.reduce_sum(mask, axis=-1), [-1, 1])

        recon = tf.losses.mean_squared_error(x, vq_x)
        vq = tf.losses.mean_squared_error(x_decode,tf.stop_gradient(x_encode))
        commit = tf.losses.mean_squared_error(x_encode,tf.stop_gradient(x_decode))

        vqvae_loss = recon + vq + self.beta * commit

        return vq_mean, vq_x, x_encode, x_decode, vqvae_loss


    def sampled_softmax_loss(self, x, y):
        with tf.name_scope('sample_softmax_loss'):

            y = tf.reshape(y, [-1, 1])

            loss = tf.nn.sampled_softmax_loss(
                weights=self.item_embedding_table,
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
        flattened = tf.reshape(x, [-1, self.code_book_dim])
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
    
    def l2(self, a,b,dim):
        a = tf.reshape(a, [-1, dim])
        b = tf.reshape(b, [-1, dim])
        distances = (
            tf.reduce_sum(a ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(b ** 2, axis=1)
            - 2 * tf.matmul(a, b, transpose_b=True)
        )
        return distances

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
        self.beta = 0.25
        self.loss = self.create_forward_path(args, name)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        
    def create_forward_path(self, args, name='DNN'):
        loss = 0
        with tf.variable_scope(name, tf.AUTO_REUSE):
            mask = tf.cast(tf.greater_equal(self.history_item_masks, 1), tf.float32)
            
            # get embedding
            histroy_item_embeddings = tf.nn.embedding_lookup(self.item_embedding_table, self.history_item_ids)
            next_item_embeddings = tf.nn.embedding_lookup(self.item_embedding_table, self.item_ids)

            self.upperboundary_tf = tf.reduce_max(histroy_item_embeddings)
            self.lowerboundary_tf = tf.reduce_min(histroy_item_embeddings)
            self.stddev_tf = tf.math.reduce_std(histroy_item_embeddings)

            # histroy_item_embeddings = tf.nn.dropout(histroy_item_embeddings ,rate=args.dropout)
            histroy_item_embeddings *= tf.reshape(mask, (-1, self.max_len, 1))

            mask = tf.reshape(mask, [-1,  self.max_len])
            masks = tf.concat([tf.expand_dims(mask, -1) for _ in range(self.embedding_dim)], axis=-1)

            histroy_item_embeddings = tf.reshape(histroy_item_embeddings, [-1, self.max_len, self.embedding_dim])
            self.histroy_item_embeddings_mean = tf.reduce_sum(histroy_item_embeddings, 1) / (tf.reduce_sum(tf.cast(masks, dtype=tf.float32), 1) + 1e-9)

            # use vqvae
            if args.vqvae:

                x_meta,_ = self.get_quantized(histroy_item_embeddings, self.item_book)

                vqvae_item_embeddings_mean, vq_x, x_encode, x_decode,  self.vq_loss = self.vqvae(histroy_item_embeddings, self.user_book, mask)
                loss += self.vq_loss

                x_encode_meta,_ = self.encoder(x_meta)
                x_encode_meta = tf.layers.max_pooling2d(x_encode_meta,pool_size=(1,2),strides=1)
                
                x_encode = tf.layers.max_pooling2d(x_encode,pool_size=(1,2),strides=1)
                x_decode = tf.layers.max_pooling2d(x_decode,pool_size=(1,2),strides=1)

                x_encode  = tf.reshape(x_encode, [self.batch_size, self.code_book_dim])
                x_decode  = tf.reshape(x_decode, [self.batch_size, self.code_book_dim])
                x_encode_meta  = tf.reshape(x_encode_meta, [self.batch_size, self.code_book_dim])
                histroy_item_embeddings_mean = tf.concat([x_encode, x_decode, x_encode_meta], axis=-1)

                if args.ISCS:

                    X = tf.tile(x_encode, [1, self.embedding_num])
                    X = tf.reshape(X, [-1, self.code_book_dim])
                    Z = tf.tile(self.user_book, [self.batch_size, 1])
                    XZ = tf.concat([X,Z], axis=-1)
                    concat_mean = self.mixer(XZ, out_dim=self.embedding_dim, name=0)
                    concat_mean = tf.reshape(concat_mean, [-1, self.embedding_num*self.embedding_dim])# bs, code_num*dim

                    dist = self.l2(x_encode, self.user_book, dim=self.code_book_dim)
                    p = tf.nn.softmax(dist, axis=1) + 1e-12
                    p = 1/p # bs, code_book_num
                    IS = p/tf.reduce_sum(p, axis=1, keepdims=True)

                    # center = tf.reduce_mean(x_encode, axis=0, keepdims=True)
                    dist = self.l2(x_encode, x_encode, dim=self.code_book_dim)
                    dist = tf.reduce_sum(dist, axis=1, keepdims=True)
                    p = tf.nn.softmax(dist, axis=0) + 1e-12
                    p = tf.reshape(1-p, [1, -1]) # 1, bs
                    CS = p/tf.reduce_sum(p, axis=1, keepdims=True)

                    # bs, code_book_num
                    front_door = tf.reshape(CS@concat_mean, [self.embedding_num, self.embedding_dim])
                    front_door = IS @ front_door

                    a = tf.losses.mean_squared_error(front_door, next_item_embeddings)
                    # b = self.sampled_softmax_loss(front_door, self.item_ids)
                    self.front_loss = a
                    loss += self.front_loss

                    histroy_item_embeddings_mean = tf.concat([x_encode, x_decode, x_encode_meta], axis=-1)

                self.histroy_item_embeddings_mean = self.mixer(histroy_item_embeddings_mean, out_dim=self.embedding_dim, name=1)

            else:
                histroy_item_embeddings_mean = tf.reduce_sum(histroy_item_embeddings, 1) / (tf.reduce_sum(tf.cast(masks, dtype=tf.float32), 1) + 1e-9)
                self.histroy_item_embeddings_mean = self.mixer(histroy_item_embeddings_mean, out_dim=self.embedding_dim, name=0)

            # get loss
            self.ce_loss = self.sampled_softmax_loss(self.histroy_item_embeddings_mean, self.item_ids)
            loss += self.ce_loss

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

        loss, vq_loss, code_book, _ , upperboundary, lowerboundary, stddev = sess.run(
            [self.loss, self.vq_loss, self.item_embedding_table, self.opt, self.upperboundary_tf, self.lowerboundary_tf, self.stddev_tf],
            feed_dict=feed_dict
        )

        self.upperboundary = max(upperboundary, self.upperboundary)
        self.lowerboundary = min(lowerboundary, self.lowerboundary)
        self.stddev = max(stddev, self.stddev)

        return loss, vq_loss, code_book
