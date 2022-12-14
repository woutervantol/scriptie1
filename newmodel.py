import numpy as np
from constants import *
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

model = keras.models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(int(num_pwfs_pixels/2), int(num_pwfs_pixels/2), 4), data_format="channels_last"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation="relu", strides=2, padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(4, (1, 1), padding="same"))
model.add(layers.BatchNormalization())
# model.add(layers.Dropout(0.4))    
model.add(layers.Flatten())
model.add(layers.Dense(400, activation="linear", kernel_regularizer=keras.regularizers.L1(l1=1e-6)))
model.summary()


# inputs = keras.Input(shape=(num_pwfs_pixels**2))
# outputs = layers.Dense(num_actuators_across_pupil**2)(inputs)
# model = keras.Model(inputs=inputs, outputs=outputs, name="pwfs_model")
# model.summary()

datamatrix = np.load("./data/random_noise/datamatrix.npy")
labelmatrix = np.load("./data/random_noise/labelmatrix.npy")

datavars = np.sqrt(np.var(datamatrix, axis=1))
labelvars = np.sqrt(np.var(labelmatrix, axis=1))
data = datamatrix / datavars[:,None]
labels = labelmatrix / labelvars[:,None]


model.compile(
    loss=keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[keras.metrics.MeanSquaredError()],
)

# oldmodel = keras.models.load_model("./models/testmodel")
# keras.backend.set_value(oldmodel.optimizer.learning_rate, 0.00001)

callback = keras.callbacks.EarlyStopping(monitor="val_loss", verbose=1, patience=10)

# history = model.fit(data.reshape(len(data), int(num_pwfs_pixels/2), int(num_pwfs_pixels/2), 4), labels, epochs=400, batch_size=64, shuffle=True, validation_split=0.2, callbacks=[callback])
history = model.fit(data.reshape(len(data), int(num_pwfs_pixels/2), int(num_pwfs_pixels/2), 4), labels, epochs=50, batch_size=64, shuffle=True)
print(history.params)
model.save("./models/testmodel")

