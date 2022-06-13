def vqvae(self, x, code_book, mask=None):
        # vqvae
        with tf.name_scope('vqvae'):

            if self.args.ISCS:
                CS = self.CS(x)
                IS = self.IS(x, code_book)
                CS_ = tf.reshape(CS, [-1, self.embedding_dim])
                IS_ = tf.reshape(IS, [-1, self.embedding_dim])
                
                ISCS = tf.concat([IS_, CS_], axis=-1)

                vq_x = tf.layers.dense(ISCS, self.embedding_dim, activation=None, name='mix_CS_and_IS')
                vq_x = tf.contrib.layers.layer_norm(vq_x)
            else:
                vq_x, encodings = self.get_quantized(x, code_book)

            with tf.variable_scope('vqvae_decoder'):
                # 维度变换
                x_ = ResBlock(vq_x, self.embedding_dim)
                x_ = ResBlock(x_, self.embedding_dim)
                x_ = tf.nn.relu(x_, 'de_relu0')
                x_ = tf.layers.conv2d_transpose(x_, filters=self.embedding_dim//2, kernel_size=(4,4), strides=(2,2), name='decnn0')
                x_ = tf.layers.batch_normalization(x_, name='de_bn0')
                x_ = tf.nn.relu(x_, 'de_relu1')
                x_ = tf.layers.conv2d_transpose(x_, filters=1, kernel_size=(6,6), strides=(2,2), name='decnn1')
                x_ = tf.nn.tanh(x_)

                x_ = tf.transpose(x_, [0,3,1,2]) #put channel to last
                x_ = tf.squeeze(x_, axis=1)
            # vq_x = self.encoder(vq_x)
            vq_x = tf.reshape(x, [-1, self.max_len, self.embedding_dim])
            vq_mean = tf.reduce_sum(vq_x, 1) / tf.reshape(tf.reduce_sum(mask, axis=-1), [-1, 1])
            
            # loss
            # commitment_loss = tf.reduce_sum(tf.reduce_sum((tf.stop_gradient(vq_x) - x) ** 2, axis=-1) * mask, axis=-1) / tf.reduce_sum(mask, axis=-1)
            # commitment_loss = tf.reduce_mean(commitment_loss)

            # codebook_loss = tf.reduce_sum(tf.reduce_sum((vq_x - tf.stop_gradient(x)) ** 2, axis=-1) * mask , axis=-1) / tf.reduce_sum(mask, axis=-1)
            # codebook_loss = tf.reduce_mean(codebook_loss)
            # vqvae_loss = 0.25*commitment_loss + codebook_loss

            #vqvae 每个z之间只跟自己有self attn 最优
            # q,k,v = self.qkv(self.code_book, self.code_book, self.code_book, dim=self.embedding_dim)
            # # q = self.code_book
            # # k = self.code_book
            # logits = q @ tf.transpose(k) / (self.embedding_dim ** 0.5)
            # # a = tf.nn.softmax(qk, -1)
            # label = tf.eye(num_rows=self.embedding_num, num_columns=self.embedding_num)
            # vqvae_loss = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=logits)

        return vq_mean

    
# def vqvae_encoder()
def ResBlock(x, dim):
    shortcut = tf.nn.relu(x)
    shortcut = tf.layers.conv2d(shortcut, filters=dim, kernel_size=(3,3))
    shortcut = tf.layers.batch_normalization(shortcut)
    shortcut = tf.nn.relu(x)
    shortcut = tf.layers.conv2d(shortcut, filters=dim, kernel_size=(1,1))
    shortcut = tf.layers.batch_normalization(shortcut)
    
    return x+shortcut


with tf.variable_scope('vqvae_decoder'):
                # 维度变换
                x_ = ResBlock(vq_x, self.embedding_dim)
                x_ = ResBlock(x_, self.embedding_dim)
                x_ = tf.nn.relu(x_, 'de_relu0')
                x_ = tf.layers.conv2d_transpose(x_, filters=self.embedding_dim//2, kernel_size=(4,4), strides=(2,2), name='decnn0')
                x_ = tf.layers.batch_normalization(x_, name='de_bn0')
                x_ = tf.nn.relu(x_, 'de_relu1')
                x_ = tf.layers.conv2d_transpose(x_, filters=1, kernel_size=(6,6), strides=(2,2), name='decnn1')
                x_ = tf.nn.tanh(x_)

                x_ = tf.transpose(x_, [0,3,1,2]) #put channel to last
                x_ = tf.squeeze(x_, axis=1)