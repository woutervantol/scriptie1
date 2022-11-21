from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
import time
import os
from constants import *


deformable_mirror = DeformableMirror(influence_functions)
num_modes = deformable_mirror.num_actuators

pwfs = PyramidWavefrontSensorOptics(pupil_grid, pwfs_grid, separation=1.2*pupil_grid_diameter, pupil_diameter=telescope_diameter, wavelength_0=wavelength_wfs, q=4)
# pwfs = ModulatedPyramidWavefrontSensorOptics()
wf = Wavefront(aperture, wavelength_wfs)


# reconstruction_matrix = np.load("./data/reconstructionMatrix001.npy")
# image_ref = Field(np.load("./data/image_ref.npy"), pwfs_grid)
# inputs = np.linspace(-10, 10, 50)
# outputs = []
# for i in inputs:
#     amplitude = probe_amp = -i * wavelength_wfs /(2*np.pi)
#     amps = np.zeros((num_actuators_across_pupil**2,))
#     amps[int((num_actuators_across_pupil**2)/2) - 5] = amplitude
#     deformable_mirror.actuators = amps
#     dm_wf = deformable_mirror.forward(wf)
#     wfs_wf = pwfs.forward(dm_wf)
#     image = wfs_wf.intensity
#     image /= np.sum(image)
#     matrixprediction = np.matmul(reconstruction_matrix, image - image_ref)
#     # deformable_mirror.actuators = matrixprediction
#     # plaatje = deformable_mirror.opd
#     # plaatje[aperture == 0] = 0
#     # plt.imshow(plaatje.reshape(256, 256))
#     # plt.show()
#     output = matrixprediction[int((num_actuators_across_pupil**2)/2)-5] / wavelength_wfs *(2*np.pi)
#     outputs.append(output)
#     # plt.imshow(matrixprediction.reshape(20, 20))
#     # plt.show()

# plt.plot(inputs, outputs)
# plt.plot(np.linspace(-2, 2, 10), -np.linspace(-2, 2, 10), ls="dashed", lw=1)
# plt.show()
# dadassadadssad


camera = NoiselessDetector(pwfs_grid)
camera.integrate(pwfs.forward(wf), 1)
image_ref = camera.read_out()
image_ref /= image_ref.sum()
# np.save("./data/image_ref", image_ref)
# imshow_field(image_ref)
# plt.show()


def makeMatrix():
    probe_amp = 0.01 * wavelength_wfs /(2*np.pi) #in rad
    slopes = []

    for ind in range(num_modes):
        if ind % 10 == 0:
            print("Measure response to mode {:d} / {:d}".format(ind+1, num_modes))
        slope = 0

        for s in [1, -1]:
            amp = np.zeros((num_modes,))
            amp[ind] = s * probe_amp
            deformable_mirror.actuators = amp
            dm_wf = deformable_mirror.forward(wf)
            wfs_wf = pwfs.forward(dm_wf)

            # camera.integrate(wfs_wf, 1)
            # image = camera.read_out()
            image = wfs_wf.intensity
            image /= np.sum(image)

            slope += s * (image-image_ref)/(2 * probe_amp)

        slopes.append(slope)
    slopes = ModeBasis(slopes)
    rcond = 1E-15
    matrix = inverse_tikhonov(slopes.transformation_matrix, rcond=rcond, svd=None)

    np.save("./data/reconstructionMatrix", matrix)
    print(matrix.shape)
    return

# makeMatrix()

# reconstruction_matrix = np.load("./data/reconstructionMatrix.npy")

spatial_resolution = wavelength_wfs / telescope_diameter
focal_grid = make_focal_grid(q=8, num_airy=20, spatial_resolution=spatial_resolution)
prop = FraunhoferPropagator(pupil_grid, focal_grid)
norm = prop(wf).power.max()



def perlin_noise(rms):
    amplitude = wavelength_wfs*rms/(2*np.pi)
    noisefunc = PerlinNoise(octaves=5, seed=0)
    xpix, ypix = num_actuators_across_pupil, num_actuators_across_pupil
    noise = np.array([[noisefunc([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)]).flatten()
    noise -= (np.max(noise) + np.min(noise))/2
    noise = noise / np.max(noise) * amplitude
    # deformable_mirror.actuators = noise
    return noise

def random_noise(rms):
    amplitude = rms*wavelength_wfs/(2*np.pi)
    noise = np.random.randn(num_actuators_across_pupil, num_actuators_across_pupil).flatten() * amplitude
    return noise



def makeData(rms):
    deformable_mirror.actuators = random_noise(rms)

    dm_wf = deformable_mirror.forward(wf)
    pwfs_wf = pwfs.forward(dm_wf)

    image = pwfs_wf.intensity
    image /= np.sum(image)
    nr_photons = 1e6
    image = np.random.poisson(image*nr_photons)
    image_ref = image/np.sum(image)
    return image_ref.reshape(num_pwfs_pixels, num_pwfs_pixels), deformable_mirror.actuators


    # imshow_field(image_ref)
    # plt.colorbar()
    # plt.show()

    # plt.imshow(deformable_mirror.actuators.reshape(num_actuators_across_pupil, num_actuators_across_pupil), cmap="gray")
    # plt.colorbar()
    # plt.show()

    # PSF_in = prop(deformable_mirror.forward(wf)).power

    # imshow_psf(PSF_in / norm, vmax=1, vmin=1e-5, spatial_resolution=spatial_resolution)
    # plt.show()


datamatrix = np.ndarray((nr_runs*len(rmslist), int(num_pwfs_pixels/2)**2, 4))
labelmatrix = np.ndarray((nr_runs*len(rmslist), num_actuators_across_pupil**2))


    # os.mkdir("./Data/random_noise/measurements/rms_{}rad".format(i))
    # os.mkdir("./Data/random_noise/dm_states/rms_{}rad".format(i))
for run in range(nr_runs):
    for rms_idx in range(len(rmslist)):
        measurement, dm_state = makeData(rmslist[rms_idx])
        rowposition = rms_idx * nr_runs + run
        labelmatrix[rowposition] = dm_state
        h = int(num_pwfs_pixels/2)
        datamatrix[rowposition,:,0] = measurement[:h, :h].flatten()
        datamatrix[rowposition,:,1] = measurement[h:, :h].flatten()
        datamatrix[rowposition,:,2] = measurement[:h, h:].flatten()
        datamatrix[rowposition,:,3] = measurement[h:, h:].flatten()
    print(run, nr_runs)
np.save("./data/random_noise/valx", datamatrix)
np.save("./data/random_noise/valy", labelmatrix)
        # np.save("./Data/random_noise/measurements/rms_{}rad/random_test{}".format(i, j), measurement)
        # np.save("./Data/random_noise/dm_states/rms_{}rad/random_test{}".format(i, j), dm_state)


