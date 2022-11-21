from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from constants import *

def test_per_image():
    for i in range(len(rmslist)):
        dm_state = labelmatrix[i*nr_runs]
        measurement = (datamatrix[i*nr_runs]/np.sqrt(np.var(datamatrix[i*nr_runs]))).reshape(64, 64, 4)[None, :]
        cnn_pred = model.predict(measurement)[0] * np.sqrt(np.var(dm_state))

        print(dm_state[:10])
        plt.imshow(dm_state.reshape(20, 20))
        plt.colorbar()
        plt.title("label")
        plt.show()
        plt.imshow((dm_state-cnn_pred).reshape(20, 20))
        plt.colorbar()
        plt.title("diff")
        plt.show()
        print(cnn_pred[:10])
        plt.imshow(cnn_pred.reshape(20, 20))
        plt.colorbar()
        plt.title("prediction")
        plt.show()

#import data and stuff
image_ref = Field(np.load("./data/image_ref.npy"), pwfs_grid)
matrix = np.load("./data/reconstructionMatrix.npy")

datamatrix = np.load("./data/random_noise/testx.npy")
labelmatrix = np.load("./data/random_noise/testy.npy")

transf_matrix = influence_functions.transformation_matrix.toarray().transpose()*2
opd_dm_states = np.matmul(labelmatrix, transf_matrix)

#import model
model = keras.models.load_model("./models/testmodel")
model.summary()

# test_per_image()


#normalize data and predict actuator states
data_variances = np.sqrt(np.var(datamatrix, axis=1))
label_variances = np.sqrt(np.var(labelmatrix, axis=1))
cnnpredictions = model.predict((datamatrix / data_variances[:,None]).reshape(len(datamatrix), int(num_pwfs_pixels/2), int(num_pwfs_pixels/2), 4)) * label_variances[:,None]
print(cnnpredictions)
print(labelmatrix)
opd_cnnpredictions = np.matmul(cnnpredictions, transf_matrix)

#reshape 4 seperate images from pyramid into 1 image
measurements = np.zeros((len(datamatrix), num_pwfs_pixels, num_pwfs_pixels))
h = int(num_pwfs_pixels/2)
measurements[:, :h, :h] = datamatrix[:, :, 0].reshape(len(datamatrix), h, h)
measurements[:, h:, :h] = datamatrix[:, :, 1].reshape(len(datamatrix), h, h)
measurements[:, :h, h:] = datamatrix[:, :, 2].reshape(len(datamatrix), h, h)
measurements[:, h:, h:] = datamatrix[:, :, 3].reshape(len(datamatrix), h, h)
measurements = measurements.reshape(len(datamatrix), num_pwfs_pixels**2)

#predict actuator states linearly
matrixpredictions = np.matmul(measurements - image_ref[None,:], matrix.transpose())
opd_matrixpredictions = np.matmul(matrixpredictions, transf_matrix)

opd_dm_states *= aperture
opd_cnnpredictions *= aperture
opd_matrixpredictions *= aperture

#calculate RMS, note that I devide by sum of aperture instead of length of aperture since the edges are often a float between 1 and 0 instead of an int
inputs = []
matrix_outputs = []
cnn_outputs = []
for i in range(len(rmslist)):
    print(i)
    input_rms = np.sqrt(np.sum(opd_dm_states[i*nr_runs:(i+1)*nr_runs]**2, axis=1) / np.sum(aperture)) / wavelength_wfs * 2*np.pi
    matrix_rms = np.sqrt(np.sum((opd_matrixpredictions[i*nr_runs:(i+1)*nr_runs] - opd_dm_states[i*nr_runs:(i+1)*nr_runs])**2, axis=1) / np.sum(aperture)) / wavelength_wfs * 2*np.pi
    cnn_rms = np.sqrt(np.sum((opd_cnnpredictions[i*nr_runs:(i+1)*nr_runs] - opd_dm_states[i*nr_runs:(i+1)*nr_runs])**2, axis=1) / np.sum(aperture)) / wavelength_wfs * 2*np.pi
    
    inputs.append(np.mean(input_rms))
    matrix_outputs.append(np.mean(matrix_rms/input_rms))
    cnn_outputs.append(np.mean(cnn_rms/input_rms))
plt.semilogx(inputs, matrix_outputs, marker="o", color="blue", label="matrix")
plt.xlabel("input RMS(rad)")
plt.ylabel("residual RMS / input RMS")
plt.semilogx(inputs, cnn_outputs, marker="o", color="green", label="cnn")
plt.xlabel("input RMS(rad)")
plt.ylabel("residual RMS / input RMS")
plt.show()



#for testing and plotting
def test_per_image():
    for i in range(len(datamatrix)):
        dm_state = labelmatrix[i]
        cnn_pred = model.predict(datamatrix[i])

        print(dm_state)
        plt.imshow(dm_state.reshape(20, 20))
        plt.colorbar()
        plt.show()
        print(cnnprediction)
        plt.imshow(cnnprediction.reshape(20, 20))
        plt.colorbar()
        plt.show()
