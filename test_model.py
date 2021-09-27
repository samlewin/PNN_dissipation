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
drhobar = -4449.54142238117

def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)

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

#Load pre-trained model with custom evaluation metrics
model = keras.models.load_model(run_dir + 'model_'+model_name+'.h5', 
                                custom_objects={'negative_loglikelihood': negative_loglikelihood,
                                                'mean': mean,
                                                'mse': mse,
                                                'var': var,
                                                'moment3': moment3,
                                                'moment4': moment4})
def test_timestep(T):
    '''Function for making predictions on vertical slice of test data at timestep T.
    Saves a single sample of predictions over the slice, as well as an ensemble of 100 predictions for the first column.'''
    loc = run_dir+'test_data/' + T
    test_dudz = np.load(loc+'/'+'test_dudz.npy')
    test_dvdz = np.load(loc+'/'+'test_dvdz.npy')
    test_drdz = np.load(loc+'/'+'test_drdz.npy')
    test_data   = np.zeros((500,500, 2))
    test_data[:,:,0] = (test_dudz**2+test_dvdz**2)/32000
    test_data[:,:,1] = (drhobar + test_drdz)/drhobar

    #Single sample of predictions
    T8_pred = model.predict(test_data)
    np.save(run_dir +model_name+'_pred_'+T, T8_pred)
    
    #Ensemble of predictions (for uncertainties)
    predictions=[]
    for i in range(100):
        predictions.append(model(test_data).sample())
    np.save(run_dir+model_name+'_ensemble_pred_' + T, np.array(predictions))
    
#Choose time steps to test on: available timesteps are T=1, T=2, T=4, T=6, T=8
timesteps = ['T=1', 'T=2', 'T=4', 'T=6', 'T=8']

for timestep in timesteps:
    test_timestep(timestep)
    #Get gradients with respect to inputs for timestep t
    t = timestep
    loc = run_dir + 'test_data/' + t
    test_dudz = np.load(loc+'/'+'test_dudz.npy')
    test_dvdz = np.load(loc+'/'+'test_dvdz.npy')
    test_drdz = np.load(loc+'/'+'test_drdz.npy')
    test_data   = np.zeros((500,500, 2))
    test_data[:,:,0] = (test_dudz**2+test_dvdz**2)/32000
    test_data[:,:,1] = (drhobar + test_drdz)/drhobar

    inputs = tf.cast(test_data, tf.float32)

    with tf.GradientTape() as tape:
            tape.watch(inputs)
            preds = model(inputs).mean()

    grads = tape.gradient(preds, inputs)
    np.save(run_dir+model_name+'_grads_'+t,grads)
    np.save(run_dir+model_name+'_preds_'+t,preds)


