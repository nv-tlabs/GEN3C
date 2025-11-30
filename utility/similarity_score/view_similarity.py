# Visualize the saved similarity (variance) matrix.
# Only prints out the shape and a small part of the matrix to avoid spamming the terminal.
import numpy as np

sim = np.load("similarity.npy")
print(sim.shape)
print(sim[:5, :5])
