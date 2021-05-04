import csv
import glob
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import subprocess

from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

optitrack_file = ''

frames = []

# load OptiTrack file
with open(optitrack_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
        if line_count < 7:
            line_count += 1
            continue

        num_elements = int((len(row)-2)/3)
        markers = []

        for v in range(num_elements):
            if row[3*v+2] not in (None, ''):
                markers.append(np.asarray([float(value) for value in row[3*v+2:3*v+5]]))

        if len(markers) > 21:
            markers = []

        frames.append(markers)

x = np.arange(len(frames))
y = np.array([len(frame) for frame in frames])

plt.plot(x, y)
plt.show()
