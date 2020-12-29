from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)


from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error

'''
#GPU disable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

'''




import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'



'''

#because of below code GPU start working
import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)

config.gpu_options.per_process_gpu_memory_fraction = 0.2
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

#Above 5 lines of code for GPU 

'''




# fit MLP to dataset and print error
def fit_model(X, y):
	# design network
	model = Sequential()
	model.add(Dense(10, input_dim=1))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	# fit network
	model.fit(X, y, epochs=100, batch_size=len(X), verbose=0)
	# forecast
	yhat = model.predict(X, verbose=0)
	print(mean_squared_error(y, yhat[:,0]))

# create sequence
length = 10
sequence = [i/float(length) for i in range(length)]
# create X/y pairs
df = DataFrame(sequence)
df = concat([df.shift(1), df], axis=1)
df.dropna(inplace=True)
# convert to MLP friendly format
values = df.values
X, y = values[:,0], values[:,1]
# repeat experiment
repeats = 10
for _ in range(repeats):
	fit_model(X, y)
