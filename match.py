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

CONST_INTERP = 5

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
frame_seed_ordered[0] = frame_seed[5]
frame_seed_ordered[1] = frame_seed[18]
frame_seed_ordered[2] = frame_seed[17]
frame_seed_ordered[3] = frame_seed[19]
frame_seed_ordered[4] = frame_seed[20]
frame_seed_ordered[5] = frame_seed[0]
frame_seed_ordered[6] = frame_seed[1]
frame_seed_ordered[7] = frame_seed[10]
frame_seed_ordered[8] = frame_seed[8]
frame_seed_ordered[9] = frame_seed[3]
frame_seed_ordered[10] = frame_seed[9]
frame_seed_ordered[11] = frame_seed[4]
frame_seed_ordered[12] = frame_seed[16]
frame_seed_ordered[13] = frame_seed[12]
frame_seed_ordered[14] = frame_seed[11]
frame_seed_ordered[15] = frame_seed[2]
frame_seed_ordered[16] = frame_seed[13]
frame_seed_ordered[17] = frame_seed[7]
frame_seed_ordered[18] = frame_seed[6]
frame_seed_ordered[19] = frame_seed[15]
frame_seed_ordered[20] = frame_seed[14]

frames[f_seed] = frame_seed_ordered

interpolated = np.full((21, len(frames)), 0, dtype=bool)
interpolated[:,f_seed] = False

print('past matching')

for f in tqdm(range(f_seed-1, -1, -1)):
    frame_ref = frames[f + 1]
    frame_cur = frames[f]

    matches = [None] * len(frame_ref)

    matrix_skip = np.full((len(frame_cur), len(frame_ref)), False, dtype=bool)
    matrix_dist = np.zeros((len(frame_cur), len(frame_ref)))

    for i in range(len(frame_cur)):
        for j in range(len(frame_ref)):
            if frame_ref[j] is None:
                matrix_skip[i, j] = True
                matrix_dist[i, j] = float('inf')
            else:
                matrix_dist[i, j] = np.linalg.norm(frame_ref[j] - frame_cur[i])

    for m in range(len(frame_cur)):
        index_min = (-1, -1)
        dist_min = float('inf')

        for i in range(len(frame_cur)):
            for j in range(len(frame_ref)):
                if matrix_skip[i, j]:
                    continue

                if matrix_dist[i, j] < dist_min:
                    index_min = (i, j)
                    dist_min = matrix_dist[i, j]

        matrix_skip[index_min[0],:] = True
        matrix_skip[:,index_min[1]] = True
        matches[index_min[1]] = index_min[0]

    frame_new = [None] * len(frame_ref)

    for m in range(len(frame_ref)):
        if matches[m] is not None:
            frame_new[m] = frame_cur[matches[m]]
        else:
            #frame_new[m] = frames[f + 1][m] + (frames[f + 1][m] - frames[f + 1 + CONST_INTERP][m]) / CONST_INTERP
            frame_new[m] = frames[f + 1][m]
            interpolated[m,f] = True

    frames[f] = frame_new

print('future matching')

for f in tqdm(range(f_seed+1, len(frames))):
    frame_ref = frames[f - 1]
    frame_cur = frames[f]

    matches = [None] * len(frame_ref)

    matrix_skip = np.full((len(frame_cur), len(frame_ref)), False, dtype=bool)
    matrix_dist = np.zeros((len(frame_cur), len(frame_ref)))

    for i in range(len(frame_cur)):
        for j in range(len(frame_ref)):
            if frame_ref[j] is None:
                matrix_skip[i, j] = True
                matrix_dist[i, j] = float('inf')
            else:
                matrix_dist[i, j] = np.linalg.norm(frame_ref[j] - frame_cur[i])

    for m in range(len(frame_cur)):
        index_min = (-1, -1)
        dist_min = float('inf')

        for i in range(len(frame_cur)):
            for j in range(len(frame_ref)):
                if matrix_skip[i, j]:
                    continue

                if matrix_dist[i, j] < dist_min:
                    index_min = (i, j)
                    dist_min = matrix_dist[i, j]

        matrix_skip[index_min[0],:] = True
        matrix_skip[:,index_min[1]] = True
        matches[index_min[1]] = index_min[0]

    frame_new = [None] * len(frame_ref)

    for m in range(len(frame_ref)):
        if matches[m] is not None:
            frame_new[m] = frame_cur[matches[m]]
        else:
            #frame_new[m] = frames[f - 1][m] + (frames[f - 1][m] - frames[f - 1 - CONST_INTERP][m]) / CONST_INTERP
            frame_new[m] = frames[f - 1][m]
            interpolated[m,f] = True

    frames[f] = frame_new

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

print('interpolating')

last_tracks = [-1] * 21

for f in tqdm(range(len(frames))):
    for m in range(21):
        if not interpolated[m,f]:
            if last_tracks[m] != -1 and last_tracks[m] != f - 1:
                line_full = frames[f][m] - frames[last_tracks[m]][m]
                line_each = line_full / (f - last_tracks[m])

                for f_in in range(f - last_tracks[m] - 1):
                    frames[last_tracks[m] + 1 + f_in][m] = frames[last_tracks[m]][m] + f_in * line_each

            last_tracks[m] = f

print('saving frames')

for i, f in enumerate(output):
temp = []

for m in f:
    temp.append(m[0])
    temp.append(m[1])
    temp.append(m[2])

output[i] = temp

with open("matched.csv", "w", newline="") as f:
writer = csv.writer(f)
writer.writerows(output)
