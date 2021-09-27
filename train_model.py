import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
tfd = tfp.distributions
import datetime

#Location where data and model is to be stored
run_dir = ''

#GPU check

print(os.getcwd())
gpus = tf.config.list_physical_devices(device_type='GPU')
print("Num GPUs:", len(gpus))

# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True) 

model_name = 'shearr_mixed_all'
learning_rate = 0.005
drhobar = -4449.54142238117

#Define evaluations metrics and loss function

def mse(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)

def tf_moment(x, n):
    return tf.reduce_mean((x - tf.reduce_mean(x))**n)

def mean(y_true, y_pred):
    return tf.reduce_mean(y_pred)/tf.reduce_mean(y_true)

def var(y_true, y_pred):
    return tf_moment(y_pred,2)/tf_moment(y_true,2)

def moment3(y_true, y_pred):
    return tf_moment(y_pred,3)/tf_moment(y_true,3)

def moment4(y_true, y_pred):
    return tf_moment(y_pred,4)/tf_moment(y_true,4)

def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)

#Build model

def get_compiled_model():
    inputs = keras.Input(shape=((500,2,1,)),name='inp');
    features = inputs
    features = keras.layers.Conv2D(32, (3, 1), activation='relu', input_shape=(500, 2, 1))(features)
    features = keras.layers.Conv2D(32, (3, 2), activation='relu')(features)
    features = keras.layers.MaxPooling2D((2, 1))(features)
    features = keras.layers.Conv2D(32, (3, 1), activation='relu')(features)
    features = keras.layers.Flatten()(features)
    params = keras.layers.Dense(500+500, activation="elu")(features)
    outputs= tfp.layers.IndependentNormal((500,1))(params)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss=negative_loglikelihood,
        metrics=[mse, mean, var, moment3, moment4]
    )
    return model

def get_dataset(loc):
    '''Build the training dataset. loc is the directory training data is stored in'''
    input1 = np.load(loc + 'train_dudz.npy')**2 + np.load(loc + 'train_dvdz.npy')**2
    input2 = drhobar + np.load(loc + 'train_drdz.npy')

    train_data   = np.zeros((input1.shape[0], input1.shape[1], 2))
    train_labels = np.zeros((input1.shape[0], input1.shape[1], 1))
    
    #Normalize data 
    train_data[:,:,0] = input1/32000
    train_data[:,:,1] = input2/drhobar
    train_labels[:,:,0] = np.log10(np.load(loc + 'train_dissipation.npy'))
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data,train_labels))

    #Batch data
    BATCH_SIZE = 1000
    SHUFFLE_BUFFER_SIZE = 1000
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True)

    return train_dataset

# Tensorboard set-up
# log_dir = run_dir+model_name+"/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#Compile model and print summary
model = get_compiled_model()
print(model.summary())

#Build datasets and train model
train_dataset = get_dataset(run_dir+'data/')
model.fit(train_dataset, epochs=30) 

#Save model
model.save(run_dir+'model_' + model_name+'.h5')

