
#Reference: Stanford Tensorflow Course, Lecture 1
# https://docs.google.com/presentation/d/1dizKPtp9hkuTwVDzoGZdYQb_61ULSsSUvaFfDFuhIc4/edit#slide=id.g1bcfa4d819_0_364

import tensorflow as tf

#Create Tensor
a = tf.add(3, 5)

#Create a Session
sess = tf.Session()

#Run Tensor inside Session
print(sess.run(a))

#Finalize Session
sess.close()

#Everything must be executed within a Tensorflow Session
#Sessions allocate memory

x = 2
y = 3
op1 = tf.add(x, y)
op2 = tf.multiply(x, y)
op3 = tf.pow(op2, op1)

with tf.Session() as sess:

	#op3 is a Tensor Node in the Graph
	print(op3) # Tensor("Pow:0", shape=(), dtype=int32)
	op3 = sess.run(op3) # evaluate Graph operations and assign value to op3
	print(op3) # 7776

# We can run multiple operations in a sess.run() call

# tf.Session.run(fetches, feed_dict=None, options=None, run_metadata=None)
# fetches is a list of Tensor objects

#It is possible to break a graph into chunks and parallelize

#Example:

with tf.device("/gpu:0"): # Assign to cpu:0
	a = tf.constant([1.0, 2.0], name="a")
	b = tf.constant([1.0, 2.0], name="b")
	c = tf.multiply(a, b)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print(sess.run(c))