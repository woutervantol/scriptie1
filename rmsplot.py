from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from constants import *

image_ref = Field(np.load("./data/image_ref.npy"), pwfs_grid)

# print(np.sqrt(image_ref.shape))
matrix = np.load("./data/reconstructionMatrix.npy")

model = keras.models.load_model("./models/testmodel")

model.summary()

deformable_mirror = DeformableMirror(influence_functions)

datamatrix = np.load("./data/random_noise/datamatrix.npy")
labelmatrix = np.load("./data/random_noise/labelmatrix.npy")

variances = np.sqrt(np.var(datamatrix, axis=1))
# print((datamatrix.transpose() / variances).transpose())
cnnpredictions = model.predict((datamatrix.transpose() / variances).transpose().reshape(1100, 128, 128, 1))

# fig, ax = plt.subplots()

y1 = np.zeros(len(rmslist))
y2 = np.zeros(len(rmslist))
rmss = np.zeros(len(rmslist))
x = rmslist.copy()
xcounts = np.zeros(len(rmslist))

for i in range(len(datamatrix)):
    # for j in range(100):
    measurement = datamatrix[i]
    matrixprediction = np.matmul(matrix, measurement - image_ref)
    # print(matrixprediction)
    # plt.imshow(matrixprediction.reshape(20, 20))
    # plt.colorbar()
    # plt.show()
    deformable_mirror.actuators = matrixprediction
    matrixprediction = deformable_mirror.opd

    dm_state = labelmatrix[i]
    # print(dm_state)
    # plt.imshow(dm_state.reshape(20, 20))
    # plt.colorbar()
    # plt.show()
    deformable_mirror.actuators = dm_state
    dm_state = deformable_mirror.opd

    cnnprediction = cnnpredictions[i] * np.sqrt(np.var(dm_state))
    # print(cnnprediction)
    # plt.imshow(cnnprediction.reshape(20, 20))
    # plt.colorbar()
    # plt.show()
    deformable_mirror.actuators = cnnprediction
    cnnprediction = deformable_mirror.opd

    # print(matrixprediction)
    # print(dm_state)
    # print(cnnprediction)


    cnnRMS = np.sqrt(np.sum(np.array((cnnprediction-dm_state)[aperture.nonzero()])**2)/len(dm_state[aperture.nonzero()])) / wavelength_wfs * 2*np.pi
    matrixRMS = np.sqrt(np.sum(np.array((matrixprediction-dm_state)[aperture.nonzero()])**2)/len(dm_state[aperture.nonzero()])) / wavelength_wfs * 2*np.pi
    in_rms = np.sqrt(np.sum(np.array(dm_state[aperture.nonzero()])**2)/len(dm_state[aperture.nonzero()])) / wavelength_wfs * 2*np.pi
    # print(cnnRMS, matrixRMS, in_rms)
    # plt.imshow((cnnprediction-dm_state).reshape(256, 256))
    # plt.colorbar()
    # plt.show()

    # plt.imshow(cnnprediction.reshape(256, 256))
    # plt.colorbar()
    # plt.show()

    # plt.imshow((matrixprediction).reshape(256, 256))
    # plt.colorbar()
    # plt.show()
    # print("tetetteteet")

    if i%10 == 0:
        print(i, len(datamatrix))

    y1[int(i/100)] += cnnRMS
    y2[int(i/100)] += matrixRMS
    rmss[int(i/100)] += in_rms
    xcounts[int(i/100)] += 1

print(xcounts)
y1 /= xcounts
y2 /= xcounts
rmss /= xcounts
y1 /= rmss
y2 /= rmss

plt.semilogx(x, y1, marker="o", color="green", label="cnn")
# plt.ylim(0, 1.2)
plt.xlabel("input RMS(rad)")
plt.ylabel("residual RMS / input RMS")

plt.semilogx(x, y2, marker="o", color="blue", label="matrix")
# plt.ylim(0, 1.2)
plt.xlabel("input RMS(rad)")
plt.ylabel("residual RMS / input RMS")
plt.legend()
plt.show()

    # print(dm_state)
    # plt.imshow(dm_state.reshape(20, 20))
    # plt.colorbar()
    # plt.show()
    # print(cnnprediction)
    # plt.imshow(cnnprediction.reshape(20, 20))
    # plt.colorbar()
    # plt.show()
    # print(dm_state.reshape(20, 20) - cnnprediction.reshape(20, 20))
    # plt.imshow(dm_state.reshape(20, 20) - cnnprediction.reshape(20, 20))
    # plt.colorbar()
    # plt.show()

    # deformable_mirror.actuators = matrixprediction
    # matrixprediction = deformable_mirror.opd




    # print(measurement)
    # plt.imshow(measurement.reshape(128, 128))
    # plt.colorbar()
    # plt.show()
    # print(cnnprediction)







y1 = []
y2 = []
x = []
for i in rmslist:
    cnnrmss = []
    matrixrmss = []
    inrmss = []
    for j in range(nr_runs):
        measurement = np.load("./data/random_noise/measurements/rms_{}rad/random_test{}.npy".format(i, j))
        variance = np.sqrt(np.var(measurement))
        measurement = measurement / variance
        dm_state = np.load("./data/random_noise/dm_states/rms_{}rad/random_test{}.npy".format(i, j))
        deformable_mirror.actuators = dm_state
        dm_state = deformable_mirror.opd
        
        cnnprediction = model.predict(measurement.reshape(128, 128, 1)[np.newaxis,:])
        print(measurement)
        print(cnnprediction)
        deformable_mirror.actuators = cnnprediction[0]
        cnnprediction = deformable_mirror.opd

        matrixprediction = np.matmul(matrix, measurement - image_ref)
        deformable_mirror.actuators = matrixprediction
        matrixprediction = deformable_mirror.opd
        asddas
        # imshow_field(image_ref)
        # plt.colorbar()
        # plt.show()
        # imshow_field(measurement)
        # plt.colorbar()
        # plt.show()
        # imshow_field(measurement-image_ref)
        # plt.colorbar()
        # plt.show()

        cnnRMS = np.sqrt(np.sum(np.array((cnnprediction-dm_state)[aperture.nonzero()])**2)/len(dm_state[aperture.nonzero()])) / wavelength_wfs * 2*np.pi
        matrixRMS = np.sqrt(np.sum(np.array((matrixprediction-dm_state)[aperture.nonzero()])**2)/len(dm_state[aperture.nonzero()])) / wavelength_wfs * 2*np.pi
        in_rms = np.sqrt(np.sum(np.array(dm_state[aperture.nonzero()])**2)/len(dm_state[aperture.nonzero()])) / wavelength_wfs * 2*np.pi
        
        # print(RMS)
        # print(in_rms)
        # print(RMS/in_rms)
        # print("---")

        cnnrmss.append(cnnRMS)
        matrixrmss.append(matrixRMS)
        inrmss.append(in_rms)

        
        # plt.imshow((prediction).reshape(256, 256))
        # # plt.clim(np.min(dm_state), np.max(dm_state))
        # # plt.clim(-1e-11, 1e-11)
        # plt.colorbar()
        # plt.title("RMS = {}".format(i))
        # plt.show()
        # plt.imshow(dm_state.reshape(256, 256))
        # plt.colorbar()
        # plt.title("RMS = {}".format(i))
        # plt.show()
    # plt.plot(i, np.average(points)/i)
    x.append(np.average(inrmss))
    y1.append(np.average(cnnrmss)/np.average(inrmss))
    y2.append(np.average(matrixrmss)/np.average(inrmss))
    print(x[-1], y1[-1], y2[-1])
    # print(np.average(points))
plt.semilogx(x, y1, marker="o")
# plt.ylim(0, 1.2)
plt.xlabel("input RMS(rad)")
plt.ylabel("residual RMS / input RMS")

plt.semilogx(x, y2, marker="o")
# plt.ylim(0, 1.2)
plt.xlabel("input RMS(rad)")
plt.ylabel("residual RMS / input RMS")

plt.show()
