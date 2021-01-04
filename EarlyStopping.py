# example of using early stopping
from sklearn.datasets import make_classification
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf 




if tf.test.gpu_device_name(): 
 print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
 print("Please install GPU version of TF")


#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)

with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    print(c)
#tf.compat.v1.disable_eager_execution()
#with tf.compat.v1.Session() as sess:
#    print (sess.run(c))



'''
#GPU disable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
'''



'''

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

'''

'''

#because of below code GPU start working
import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)

config.gpu_options.per_process_gpu_memory_fraction = 0.2
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

#Above 5 lines of code for GPU 

'''



















# create the dataset
X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
# determine the number of input features
n_features = X.shape[1]
# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')
# configure early stopping
es = EarlyStopping(monitor='val_loss', patience=5)
# fit the model
history = model.fit(X, y, epochs=200, batch_size=32, verbose=0, validation_split=0.3, callbacks=[es])
