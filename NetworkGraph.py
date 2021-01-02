
'''
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(2, input_dim=1, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

'''




#GPU disable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"




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






from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model


model = Sequential()
model.add(Dense(2, input_dim=1, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

