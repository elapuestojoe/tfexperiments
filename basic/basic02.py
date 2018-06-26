#Reference: Stanford Tensorflow Course, Lecture 2
# Operations
# https://docs.google.com/presentation/d/1iO_bBL_5REuDQ7RJ2F35vH2BxAiGMocLC6t_N-6eXaE/edit#slide=id.g1bd10f151e_0_0

import tensorflow as tf
a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
x = tf.add(a, b, name="add")

writer = tf.summary.FileWriter("./basic02graphs", tf.get_default_graph())

with tf.Session() as sess:
	writer = tf.summary.FileWriter("./basic02graphs", sess.graph)
	#We create a writer after graph definition and before starting a session (why?)

	print(sess.run(x))
writer.close()

#Writer created a folder where logs are saved 
#This logs can be visualized in TensorBoard using:
# tensorboard --logdir="./basic02graphs" --port 6006

#Some basic info about Variables and Operations

#Constants
	# tf.constant(
	# 	value,
	# 	dtype=None,
	# 	shape=None,
	# 	name="Const",
	# 	verify_shape=False)

#Value can be any Tensor

#Zeros
	# tf.zeros(shape, dtype=tf.float32, name=None)

#Copy Tensor shape and fill with zeroes
	# tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)
#Same but with Ones
	# tf.ones(shape, dtype=tf.float32, name=None)
	# tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)

#Fill with values
	# tf.fill(dims, value, name=None)

#Sequences
	# tf.lin_space(start, stop, num, name=None) 
		# tf.lin_space(10.0, 13.0, 4) -> [10. 11. 12. 13.]

	#tf.range(start, limit=None, delta=1, dtype=None, name="range")
		# tf.range(3, 18, 3) -> [3 6 9 12 15]

# Some Random functions
	# tf.random.normal
	# tf.truncated_normal
	# tf.random_uniform
	# tf.random_shuffle
	# tf.random_crop
	# tf.multinomial
	# tf.random_gamma

	# USEFUL FOR REPRODUCIBILITY
	# tf.set_random_seed(seed)


#Operations:

	# Math:
		# Add
		# Sub
		# Mul 
		# Div 
		# Exp 
		# Log 

	# Array:
		# Concat
		# Slice
		# Split
		# Constant 
		# Rank 

	# Matrix: 
		# MatMul 
		# MatInverse 

	# Stateful: 
		# Variable 
		# Assign 

	# Neural Network*:
		# Softmax
		# Sigmoid
		# ReLU
		# Convolution2D
		# MaxPool

	# Checkpoints:
		# Save 
		# Restore 

	# Queues:
		# Enqueue
		# Dequeue
		# MutexAcquire
		# MutexRelease

	# Control Flow:
		# Merge
		# Switch
		# Enter
		# Leave
		# NextIteration

# tf.div -> floordiv
# tf.divide -> truediv

#Constants are stored in graph, variables are not 

s = tf.get_variable("scalar", initializer=tf.constant(2))
m = tf.get_variable("matrix", initializer=tf.constant([[0, 1], [2, 3]]))
W = tf.get_variable("big_matrix", shape=(784, 10), initializer=tf.zeros_initializer())

# *Preferred over tf.Variable() initialization (faster)

# Variables must be initialized

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

# Initialize subset of variables
with tf.Session() as sess:
	sess.run(tf.variables_initializer([s, m]))

# Initialize only one variable
with tf.Session() as sess:
	sess.run(W.initializer)

W = tf.get_variable("my_matrix", shape=(10, 2), initializer=tf.truncated_normal_initializer())

with tf.Session() as sess:
	sess.run(W.initializer)
	print(W.eval()) #Tensors do not have values until evaluated


W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
	sess.run(W.initializer)
	sess.run(assign_op)			#operations must be executed
	print(W.eval())

my_var = tf.Variable(2, name="my_var")
my_var_times_two = my_var.assign(2 * my_var) #We can also assign operations to Variables

with tf.Session() as sess:
	sess.run(my_var.initializer)
	sess.run(my_var_times_two) # my_var = 4
	sess.run(my_var_times_two) # my_var = 8

	sess.run(my_var.assign_add(10)) # my_var = 18
	sess.run(my_var.assign_sub(2)) # my_var = 16


# Control dependencies:
	# g = tf.get_default_graph()
	# with g.control_dependencies([a, b, c]):
		#Everything below this will run after a b c have been executed
		# d
		# e

# Placeholders define operations to be executed without defining the Variables (Like functions)

# tf.placeholder(dtype, shape=None, name=None)
a = tf.placeholder(tf.float32, shape=[3])

b = tf.constant([5, 5, 5], tf.float32)

c = a + b 

with tf.Session() as sess:
	print(sess.run(c, feed_dict={a : [1, 2 ,3]})) 
	# We provide values for a (Could probably be loaded from a json)



# with tf.Session() as sess:
#	for a_value in list_of_values_for_a:
#	print(sess.run(c, {a: a_value}))
