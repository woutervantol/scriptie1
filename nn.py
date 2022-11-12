import numpy as np
from constants import *
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.models.Sequential()
model.add(layers.Conv2D(4, (8, 8), activation="tanh", input_shape=(num_pwfs_pixels, num_pwfs_pixels, 1), data_format="channels_last", strides=(2, 2)))
# model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(8, (8, 8), activation="tanh"))
model.add(layers.Conv2D(8, (8, 8), activation="tanh"))
model.add(layers.Conv2D(8, (8, 8), activation="tanh"))
model.add(layers.Conv2D(16, (4, 4), activation="tanh", strides=(2, 2)))
# model.add(layers.Conv2D(1, (16, 16), activation="tanh"))
model.add(layers.Flatten())
model.add(layers.Dense(400, activation="tanh"))
model.summary()


# inputs = keras.Input(shape=(num_pwfs_pixels**2))
# outputs = layers.Dense(num_actuators_across_pupil**2)(inputs)
# model = keras.Model(inputs=inputs, outputs=outputs, name="pwfs_model")
# model.summary()


datamatrix = np.load("./data/random_noise/datamatrix.npy")
labelmatrix = np.load("./data/random_noise/labelmatrix.npy")
totmatrix = np.append(datamatrix, labelmatrix, axis=1)
np.random.shuffle(totmatrix)
datamatrix = totmatrix[:, :-400]
labelmatrix = totmatrix[:, -400:]

xtrain = datamatrix[0:int(len(datamatrix)*0.8)]
xtest = datamatrix[int(len(datamatrix)*0.8):]

variances = np.sqrt(np.var(xtrain, axis=1))
xtrain = (xtrain.transpose() / variances).transpose()


ytrain = labelmatrix[0:int(len(datamatrix)*0.8)]
ytest = labelmatrix[int(len(datamatrix)*0.8):]

variances = np.sqrt(np.var(ytrain, axis=1))
ytrain = (ytrain.transpose() / variances).transpose()


model.compile(
    loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.MeanSquaredError()],
)


history = model.fit(np.reshape(xtrain, (len(xtrain), num_pwfs_pixels, num_pwfs_pixels)), ytrain, batch_size=len(rmslist), epochs=10)
print(history.params)
model.save("./models/testmodel")

