from hcipy import *
import numpy as np

wavelength_wfs = 1e-6
telescope_diameter = 39.3
zero_magnitude_flux = 3.9E10
num_pupil_pixels = 256
num_pwfs_pixels = 128
pupil_grid_diameter = telescope_diameter
pupil_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)
pwfs_grid = make_pupil_grid(num_pwfs_pixels, 1.3*2*pupil_grid_diameter)

make_aperture = make_elt_aperture()
aperture = make_aperture(pupil_grid)

num_actuators_across_pupil = 20
actuator_spacing = telescope_diameter / num_actuators_across_pupil
influence_functions = make_gaussian_influence_functions(pupil_grid, num_actuators_across_pupil, actuator_spacing)

# print(influence_functions.linear_combination())

nr_runs = 100
rmslist = np.array([0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0])
# rmslist = np.logspace(0.01, 4.0, 10)