import tensorflow as tf
import numpy as np 

dim =64
slen = 20
bs = 128
index = [
    [i for t in range(slen)] for i in range(bs)
]
print(index)
exit()

one_hot = tf.one_hot(index, depth=bs, on_value=1.0)
one_hot = tf.reshape(one_hot, [bs*slen, -1])
mask = one_hot @ tf.transpose(one_hot,[1,0])
# b = tf.ones([bs*slen, dim])
# go = mask @ b

q = tf.ones([bs*slen, dim])
k = tf.ones([bs*slen, dim])
kq = tf.matmul(k, tf.transpose(q))
test = tf.range(slen)
test = tf.expand_dims(test,axis=0)
test = tf.tile(test,[bs, 1])
one_hot = tf.one_hot(test, depth=bs, on_value=1.0)
one_hot = tf.reshape(one_hot, [bs*slen, -1])
mask = one_hot @ tf.transpose(one_hot,[1,0])

# mask= tf.one_hot(tf.range(bs), bs)

with tf.Session() as sess:
    out,kq = sess.run([mask, kq])
    print(out)
    print(np.array(out).shape)
    # print(len(out[0]))
    # print(list(range(100)))