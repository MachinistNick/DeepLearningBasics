# Example of TensorFlow library for deep learning
#This Small code will run on GPU, if you already installed it.

#Tensorflow is better than Theano for due to it's production orientation
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# declare two symbolic floating-point scalars
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# create a simple symbolic expression using the add function
add = tf.add(a, b)

# bind 1.5 to 'a', 2.5 to 'b', and evaluate 'c'
sess = tf.Session()
binding = {a: 1.5, b: 2.5}
c = sess.run(add, feed_dict=binding)
print(c)
