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
f_seed = -1

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

frame_seed = frames[f_seed]

frame_seed_ordered = [None] * len(frame_seed)
frame_seed_ordered[0] = frame_seed[0]
frame_seed_ordered[1] = frame_seed[1]
frame_seed_ordered[2] = frame_seed[2]
frame_seed_ordered[3] = frame_seed[3]
frame_seed_ordered[4] = frame_seed[4]
frame_seed_ordered[5] = frame_seed[5]
frame_seed_ordered[6] = frame_seed[6]
frame_seed_ordered[7] = frame_seed[7]
frame_seed_ordered[8] = frame_seed[8]
frame_seed_ordered[9] = frame_seed[9]
frame_seed_ordered[10] = frame_seed[10]
frame_seed_ordered[11] = frame_seed[11]
frame_seed_ordered[12] = frame_seed[12]
frame_seed_ordered[13] = frame_seed[13]
frame_seed_ordered[14] = frame_seed[14]
frame_seed_ordered[15] = frame_seed[15]
frame_seed_ordered[16] = frame_seed[16]
frame_seed_ordered[17] = frame_seed[17]
frame_seed_ordered[18] = frame_seed[18]
frame_seed_ordered[19] = frame_seed[19]
frame_seed_ordered[20] = frame_seed[20]

frames[f_seed] = frame_seed_ordered

connections = [None] * 20
connections[0] = (0, 1)
connections[1] = (1, 2)
connections[2] = (2, 3)
connections[3] = (3, 4)
connections[4] = (0, 5)
connections[5] = (5, 6)
connections[6] = (6, 7)
connections[7] = (7, 8)
connections[8] = (0, 9)
connections[9] = (9, 10)
connections[10] = (10, 11)
connections[11] = (11, 12)
connections[12] = (0, 13)
connections[13] = (13, 14)
connections[14] = (14, 15)
connections[15] = (15, 16)
connections[16] = (0, 17)
connections[17] = (17, 18)
connections[18] = (18, 19)
connections[19] = (19, 20)

plt.clf()

xs = [vector[0] for vector in frames[f_seed]]
ys = [vector[1] for vector in frames[f_seed]]
zs = [vector[2] for vector in frames[f_seed]]

center = (0.0, 0.0, 0.0)

for i in range(len(xs)):
    center = (center[0] + xs[i], center[1] + ys[i], center[2] + zs[i])

center = (center[0] / len(xs), center[1] / len(ys), center[2] / len(zs))

xs_origin = [None] * len(xs)
ys_origin = [None] * len(ys)
zs_origin = [None] * len(zs)

for i in range(len(xs)):
    xs_origin[i] = xs[i] - center[0]
    ys_origin[i] = ys[i] - center[1]
    zs_origin[i] = zs[i] - center[2]

colors = []

for m in range(21):
    colors.append('#000000')

fig = plt.figure(1)
ax = fig.add_subplot('111', projection='3d')
ax.scatter(xs_origin, ys_origin, zs_origin, c=colors)

for i, connection in enumerate(connections):
    rgb = matplotlib.colors.hsv_to_rgb([i/float(len(connections)), 1.0, 1.0])
    ax.plot([xs_origin[connection[0]], xs_origin[connection[1]]], [ys_origin[connection[0]], ys_origin[connection[1]]], [zs_origin[connection[0]], zs_origin[connection[1]]], c=rgb)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
