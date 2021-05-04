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

offset_marker = 0.75    # constant for 4mm OptiTrack markers

frames = []

with open('matched.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        num_elements = int(len(row)/3)
        markers = []

        for v in range(num_elements):
            if row[3*v] not in (None, ''):
                markers.append(np.asarray([float(value) for value in row[3*v:3*(v+1)]]))

        frames.append(markers)

# hand model constants for correct scaling
meas_o = [0.20296497971433372, 0.18095062663790207, 0.09925317008481732, 0.0966483614181743, 0.057148752257564904, 0.13558784812275593, 0.10048410314022911,
          0.08860188134604136, 0.05997917877863547, 0.13716501788178273, 0.11977307865518659, 0.08810790117097786, 0.05758307641752432, 0.1404550573292884,
          0.08865069053337495, 0.08576788649836885, 0.056389005645895135, 0.11376382586799517, 0.07928106805091102, 0.06289905745913114, 0.043744944591484335]

output = []

print('mapping')

for f in tqdm(range(len(frames))):
    plt.clf()

    xs = [vector[0] for vector in frames[f]]
    ys = [vector[1] for vector in frames[f]]
    zs = [vector[2] for vector in frames[f]]

    xs.extend([None] * len(xs))
    ys.extend([None] * len(ys))
    zs.extend([None] * len(zs))

    l_ref = np.linalg.norm(np.array([xs[9], ys[9], zs[9]]) - np.array([xs[0], ys[0], zs[0]]))
    meas = [m * l_ref for m in meas_o]

    root = np.array([xs[0], ys[0], zs[0]])
    thumb_mcp1 = np.array([xs[1], ys[1], zs[1]])
    thumb_mcp2 = np.array([xs[2], ys[2], zs[2]])
    thumb_joint3 = np.array([xs[3], ys[3], zs[3]])
    thumb_tip = np.array([xs[4], ys[4], zs[4]])
    index_mcp = np.array([xs[5], ys[5], zs[5]])
    index_joint2 = np.array([xs[6], ys[6], zs[6]])
    index_joint3 = np.array([xs[7], ys[7], zs[7]])
    index_tip = np.array([xs[8], ys[8], zs[8]])
    middle_mcp = np.array([xs[9], ys[9], zs[9]])
    middle_joint2 = np.array([xs[10], ys[10], zs[10]])
    middle_joint3 = np.array([xs[11], ys[11], zs[11]])
    middle_tip = np.array([xs[12], ys[12], zs[12]])
    ring_mcp = np.array([xs[13], ys[13], zs[13]])
    ring_joint2 = np.array([xs[14], ys[14], zs[14]])
    ring_joint3 = np.array([xs[15], ys[15], zs[15]])
    ring_tip = np.array([xs[16], ys[16], zs[16]])
    pinkie_mcp = np.array([xs[17], ys[17], zs[17]])
    pinkie_joint2 = np.array([xs[18], ys[18], zs[18]])
    pinkie_joint3 = np.array([xs[19], ys[19], zs[19]])
    pinkie_tip = np.array([xs[20], ys[20], zs[20]])

    # root
    a = middle_mcp - root
    a = a / np.linalg.norm(a)
    b = index_mcp - root
    b = b / np.linalg.norm(b)
    displacement = np.cross(a, b)
    displacement = displacement / np.linalg.norm(displacement)
    displacement = displacement * (meas[0] + offset_marker)

    root_mapped = root - displacement

    xs[21] = root_mapped[0]
    ys[21] = root_mapped[1]
    zs[21] = root_mapped[2]

    # thumb mcp 1
    a1 = thumb_mcp1 - root
    la1 = np.linalg.norm(a1)
    a1 = a1 / la1
    a2 = thumb_mcp2 - thumb_mcp1
    la2 = np.linalg.norm(a2)
    a2 = a2 / la2
    b = thumb_mcp1 - index_mcp
    b = b / np.linalg.norm(b)
    displacement1 = np.cross(a1, b)
    displacement1 = displacement1 / np.linalg.norm(displacement1)
    displacement2 = np.cross(a2, b)
    displacement2 = displacement2 / np.linalg.norm(displacement2)
    displacement = (la2 * displacement1 + la1 * displacement2) / (la1 + la2)
    displacement = displacement / np.linalg.norm(displacement)
    displacement = displacement * (meas[1] + offset_marker)

    thumb_mcp1_mapped = thumb_mcp1 - displacement

    xs[22] = thumb_mcp1_mapped[0]
    ys[22] = thumb_mcp1_mapped[1]
    zs[22] = thumb_mcp1_mapped[2]

    # thumb mcp 2
    a1 = thumb_joint3 - thumb_mcp2
    la1 = np.linalg.norm(a1)
    a1 = a1 / la1
    a2 = thumb_mcp2 - thumb_mcp1
    la2 = np.linalg.norm(a2)
    a2 = a2 / la2
    b = thumb_mcp1 - index_mcp
    displacement1 = np.cross(a1, b)
    displacement1 = displacement1 / np.linalg.norm(displacement1)
    displacement2 = np.cross(a2, b)
    displacement2 = displacement2 / np.linalg.norm(displacement2)
    displacement = (la2 * displacement1 + la1 * displacement2) / (la1 + la2)
    displacement = displacement / np.linalg.norm(displacement)
    displacement = displacement * (meas[2] + offset_marker)

    thumb_mcp2_mapped = thumb_mcp2 - displacement

    xs[23] = thumb_mcp2_mapped[0]
    ys[23] = thumb_mcp2_mapped[1]
    zs[23] = thumb_mcp2_mapped[2]

    # thumb joint 3
    a1 = thumb_tip - thumb_joint3
    la1 = np.linalg.norm(a1)
    a1 = a1 / la1
    a2 = thumb_joint3 - thumb_mcp2
    la2 = np.linalg.norm(a2)
    a2 = a2 / la2
    b = thumb_mcp1 - index_mcp
    displacement1 = np.cross(a1, b)
    displacement1 = displacement1 / np.linalg.norm(displacement1)
    displacement2 = np.cross(a2, b)
    displacement2 = displacement2 / np.linalg.norm(displacement2)
    displacement = (la2 * displacement1 + la1 * displacement2) / (la1 + la2)
    displacement = displacement / np.linalg.norm(displacement)
    displacement = displacement * (meas[3] + offset_marker)

    thumb_joint3_mapped = thumb_joint3 - displacement

    xs[24] = thumb_joint3_mapped[0]
    ys[24] = thumb_joint3_mapped[1]
    zs[24] = thumb_joint3_mapped[2]

    # thumb tip
    a = thumb_tip - thumb_joint3
    a = a / np.linalg.norm(a)
    b = thumb_mcp1 - index_mcp
    b = b / np.linalg.norm(b)
    displacement = np.cross(a, b)
    displacement = displacement / np.linalg.norm(displacement)
    displacement = displacement * (meas[4] + offset_marker)

    thumb_tip_mapped = thumb_tip - displacement

    xs[25] = thumb_tip_mapped[0]
    ys[25] = thumb_tip_mapped[1]
    zs[25] = thumb_tip_mapped[2]

    # index mcp
    a1 = index_mcp - root
    la1 = np.linalg.norm(a1)
    a1 = a1 / la1
    a2 = index_joint2 - index_mcp
    la2 = np.linalg.norm(a2)
    a2 = a2 / la2
    b = index_mcp - middle_mcp
    b = b / np.linalg.norm(b)
    displacement1 = np.cross(a1, b)
    displacement1 = displacement1 / np.linalg.norm(displacement1)
    displacement2 = np.cross(a2, b)
    displacement2 = displacement2 / np.linalg.norm(displacement2)
    displacement = (la2 * displacement1 + la1 * displacement2) / (la1 + la2)
    displacement = displacement / np.linalg.norm(displacement)
    displacement = displacement * (meas[5] + offset_marker)

    index_mcp_mapped = index_mcp - displacement

    xs[26] = index_mcp_mapped[0]
    ys[26] = index_mcp_mapped[1]
    zs[26] = index_mcp_mapped[2]

    # index joint 2
    a1 = index_joint3 - index_joint2
    la1 = np.linalg.norm(a1)
    a1 = a1 / la1
    a2 = index_joint2 - index_mcp
    la2 = np.linalg.norm(a2)
    a2 = a2 / la2
    b = index_mcp - middle_mcp
    displacement1 = np.cross(a1, b)
    displacement1 = displacement1 / np.linalg.norm(displacement1)
    displacement2 = np.cross(a2, b)
    displacement2 = displacement2 / np.linalg.norm(displacement2)
    displacement = (la2 * displacement1 + la1 * displacement2) / (la1 + la2)
    displacement = displacement / np.linalg.norm(displacement)
    displacement = displacement * (meas[6] + offset_marker)

    index_joint2_mapped = index_joint2 - displacement

    xs[27] = index_joint2_mapped[0]
    ys[27] = index_joint2_mapped[1]
    zs[27] = index_joint2_mapped[2]

    # index joint 3
    a1 = index_tip - index_joint3
    la1 = np.linalg.norm(a1)
    a1 = a1 / la1
    a2 = index_joint3 - index_joint2
    la2 = np.linalg.norm(a2)
    a2 = a2 / la2
    b = index_mcp - middle_mcp
    displacement1 = np.cross(a1, b)
    displacement1 = displacement1 / np.linalg.norm(displacement1)
    displacement2 = np.cross(a2, b)
    displacement2 = displacement2 / np.linalg.norm(displacement2)
    displacement = (la2 * displacement1 + la1 * displacement2) / (la1 + la2)
    displacement = displacement / np.linalg.norm(displacement)
    displacement = displacement * (meas[7] + offset_marker)

    index_joint3_mapped = index_joint3 - displacement

    xs[28] = index_joint3_mapped[0]
    ys[28] = index_joint3_mapped[1]
    zs[28] = index_joint3_mapped[2]

    # index tip
    a = index_tip - index_joint3
    a = a / np.linalg.norm(a)
    b = index_mcp - middle_mcp
    b = b / np.linalg.norm(b)
    displacement = np.cross(a, b)
    displacement = displacement / np.linalg.norm(displacement)
    displacement = displacement * (meas[8] + offset_marker)

    index_tip_mapped = index_tip - displacement

    xs[29] = index_tip_mapped[0]
    ys[29] = index_tip_mapped[1]
    zs[29] = index_tip_mapped[2]

    # middle mcp
    a1 = middle_mcp - root
    la1 = np.linalg.norm(a1)
    a1 = a1 / la1
    a2 = middle_joint2 - middle_mcp
    la2 = np.linalg.norm(a2)
    a2 = a2 / la2
    b1 = middle_mcp - index_mcp
    lb1 = np.linalg.norm(b1)
    b1 = b1 / lb1
    b2 = ring_mcp - middle_mcp
    lb2 = np.linalg.norm(b2)
    b2 = b2 / lb2
    b = -(lb2 * b1 + lb1 * b2) / (lb1 + lb2)
    b = b / np.linalg.norm(b)
    displacement1 = np.cross(a1, b)
    displacement1 = displacement1 / np.linalg.norm(displacement1)
    displacement2 = np.cross(a2, b)
    displacement2 = displacement2 / np.linalg.norm(displacement2)
    displacement = (la2 * displacement1 + la1 * displacement2) / (la1 + la2)
    displacement = displacement / np.linalg.norm(displacement)
    displacement = displacement * (meas[9] + offset_marker)

    middle_mcp_mapped = middle_mcp - displacement

    xs[30] = middle_mcp_mapped[0]
    ys[30] = middle_mcp_mapped[1]
    zs[30] = middle_mcp_mapped[2]

    # middle joint 2
    a1 = middle_joint3 - middle_joint2
    la1 = np.linalg.norm(a1)
    a1 = a1 / la1
    a2 = middle_joint2 - middle_mcp
    la2 = np.linalg.norm(a2)
    a2 = a2 / la2
    b1 = middle_mcp - index_mcp
    lb1 = np.linalg.norm(b1)
    b1 = b1 / lb1
    b2 = ring_mcp - middle_mcp
    lb2 = np.linalg.norm(b2)
    b2 = b2 / lb2
    b = -(lb2 * b1 + lb1 * b2) / (lb1 + lb2)
    b = b / np.linalg.norm(b)
    displacement1 = np.cross(a1, b)
    displacement1 = displacement1 / np.linalg.norm(displacement1)
    displacement2 = np.cross(a2, b)
    displacement2 = displacement2 / np.linalg.norm(displacement2)
    displacement = (la2 * displacement1 + la1 * displacement2) / (la1 + la2)
    displacement = displacement / np.linalg.norm(displacement)
    displacement = displacement * (meas[10] + offset_marker)

    middle_joint2_mapped = middle_joint2 - displacement

    xs[31] = middle_joint2_mapped[0]
    ys[31] = middle_joint2_mapped[1]
    zs[31] = middle_joint2_mapped[2]

    # middle joint 3
    a1 = middle_tip - middle_joint3
    la1 = np.linalg.norm(a1)
    a1 = a1 / la1
    a2 = middle_joint3 - middle_joint2
    la2 = np.linalg.norm(a2)
    a2 = a2 / la2
    b1 = middle_mcp - index_mcp
    lb1 = np.linalg.norm(b1)
    b1 = b1 / lb1
    b2 = ring_mcp - middle_mcp
    lb2 = np.linalg.norm(b2)
    b2 = b2 / lb2
    b = -(lb2 * b1 + lb1 * b2) / (lb1 + lb2)
    b = b / np.linalg.norm(b)
    displacement1 = np.cross(a1, b)
    displacement1 = displacement1 / np.linalg.norm(displacement1)
    displacement2 = np.cross(a2, b)
    displacement2 = displacement2 / np.linalg.norm(displacement2)
    displacement = (la2 * displacement1 + la1 * displacement2) / (la1 + la2)
    displacement = displacement / np.linalg.norm(displacement)
    displacement = displacement * (meas[11] + offset_marker)

    middle_joint3_mapped = middle_joint3 - displacement

    xs[32] = middle_joint3_mapped[0]
    ys[32] = middle_joint3_mapped[1]
    zs[32] = middle_joint3_mapped[2]

    # middle tip
    a = middle_tip - middle_joint3
    a = a / np.linalg.norm(a)
    b1 = middle_mcp - index_mcp
    lb1 = np.linalg.norm(b1)
    b1 = b1 / lb1
    b2 = ring_mcp - middle_mcp
    lb2 = np.linalg.norm(b2)
    b2 = b2 / lb2
    b = -(lb2 * b1 + lb1 * b2) / (lb1 + lb2)
    b = b / np.linalg.norm(b)
    displacement = np.cross(a, b)
    displacement = displacement / np.linalg.norm(displacement)
    displacement = displacement * (meas[12] + offset_marker)

    middle_tip_mapped = middle_tip - displacement

    xs[33] = middle_tip_mapped[0]
    ys[33] = middle_tip_mapped[1]
    zs[33] = middle_tip_mapped[2]

    # ring mcp
    a1 = ring_mcp - root
    la1 = np.linalg.norm(a1)
    a1 = a1 / la1
    a2 = ring_joint2 - ring_mcp
    la2 = np.linalg.norm(a2)
    a2 = a2 / la2
    b = middle_mcp - ring_mcp
    b = b / np.linalg.norm(b)
    displacement1 = np.cross(a1, b)
    displacement1 = displacement1 / np.linalg.norm(displacement1)
    displacement2 = np.cross(a2, b)
    displacement2 = displacement1 / np.linalg.norm(displacement2)
    displacement = (la2 * displacement1 + la1 * displacement2) / (la1 + la2)
    displacement = displacement / np.linalg.norm(displacement)
    displacement = displacement * (meas[13] + offset_marker)

    ring_mcp_mapped = ring_mcp - displacement

    xs[34] = ring_mcp_mapped[0]
    ys[34] = ring_mcp_mapped[1]
    zs[34] = ring_mcp_mapped[2]

    # ring joint 2
    a1 = ring_joint3 - ring_joint2
    la1 = np.linalg.norm(a1)
    a1 = a1 / la1
    a2 = ring_joint2 - ring_mcp
    la2 = np.linalg.norm(a2)
    a2 = a2 / la2
    b = middle_mcp - ring_mcp
    displacement1 = np.cross(a1, b)
    displacement1 = displacement1 / np.linalg.norm(displacement1)
    displacement2 = np.cross(a2, b)
    displacement2 = displacement2 / np.linalg.norm(displacement2)
    displacement = (la2 * displacement1 + la1 * displacement2) / (la1 + la2)
    displacement = displacement / np.linalg.norm(displacement)
    displacement = displacement * (meas[14] + offset_marker)

    ring_joint2_mapped = ring_joint2 - displacement

    xs[35] = ring_joint2_mapped[0]
    ys[35] = ring_joint2_mapped[1]
    zs[35] = ring_joint2_mapped[2]

    # ring joint 3
    a1 = ring_tip - ring_joint3
    la1 = np.linalg.norm(a1)
    a1 = a1 / la1
    a2 = ring_joint3 - ring_joint2
    la2 = np.linalg.norm(a2)
    a2 = a2 / la2
    b = middle_mcp - ring_mcp
    displacement1 = np.cross(a1, b)
    displacement1 = displacement1 / np.linalg.norm(displacement1)
    displacement2 = np.cross(a2, b)
    displacement2 = displacement2 / np.linalg.norm(displacement2)
    displacement = (la2 * displacement1 + la1 * displacement2) / (la1 + la2)
    displacement = displacement / np.linalg.norm(displacement)
    displacement = displacement * (meas[15] + offset_marker)

    ring_joint3_mapped = ring_joint3 - displacement

    xs[36] = ring_joint3_mapped[0]
    ys[36] = ring_joint3_mapped[1]
    zs[36] = ring_joint3_mapped[2]

    # ring tip
    a = ring_tip - ring_joint3
    a = a / np.linalg.norm(a)
    b = middle_mcp - ring_mcp
    b = b / np.linalg.norm(b)
    displacement = np.cross(a, b)
    displacement = displacement / np.linalg.norm(displacement)
    displacement = displacement * (meas[16] + offset_marker)

    ring_tip_mapped = ring_tip - displacement

    xs[37] = ring_tip_mapped[0]
    ys[37] = ring_tip_mapped[1]
    zs[37] = ring_tip_mapped[2]

    # pinkie mcp
    a1 = ring_mcp - root
    la1 = np.linalg.norm(a1)
    a1 = a1 / la1
    a2 = ring_joint2 - ring_mcp
    la2 = np.linalg.norm(a2)
    a2 = a2 / la2
    b = ring_mcp - pinkie_mcp
    b = b / np.linalg.norm(b)
    displacement1 = np.cross(a1, b)
    displacement1 = displacement1 / np.linalg.norm(displacement1)
    displacement2 = np.cross(a2, b)
    displacement2 = displacement2 / np.linalg.norm(displacement2)
    displacement = (la2 * displacement1 + la1 * displacement2) / (la1 + la2)
    displacement = displacement / np.linalg.norm(displacement)
    displacement = displacement * (meas[17] + offset_marker)

    pinkie_mcp_mapped = pinkie_mcp - displacement

    xs[38] = pinkie_mcp_mapped[0]
    ys[38] = pinkie_mcp_mapped[1]
    zs[38] = pinkie_mcp_mapped[2]

    # pinkie joint 2
    a1 = pinkie_joint3 - pinkie_joint2
    la1 = np.linalg.norm(a1)
    a1 = a1 / la1
    a2 = pinkie_joint2 - pinkie_mcp
    la2 = np.linalg.norm(a2)
    a2 = a2 / la2
    b = ring_mcp - pinkie_mcp
    displacement1 = np.cross(a1, b)
    displacement1 = displacement1 / np.linalg.norm(displacement1)
    displacement2 = np.cross(a2, b)
    displacement2 = displacement2 / np.linalg.norm(displacement2)
    displacement = (la2 * displacement1 + la1 * displacement2) / (la1 + la2)
    displacement = displacement / np.linalg.norm(displacement)
    displacement = displacement * (meas[18] + offset_marker)

    pinkie_joint2_mapped = pinkie_joint2 - displacement

    xs[39] = pinkie_joint2_mapped[0]
    ys[39] = pinkie_joint2_mapped[1]
    zs[39] = pinkie_joint2_mapped[2]

    # pinkie joint 3
    a1 = pinkie_tip - pinkie_joint3
    la1 = np.linalg.norm(a1)
    a1 = a1 / la1
    a2 = pinkie_joint3 - pinkie_joint2
    la2 = np.linalg.norm(a2)
    a2 = a2 / la2
    b = ring_mcp - pinkie_mcp
    displacement1 = np.cross(a1, b)
    displacement1 = displacement1 / np.linalg.norm(displacement1)
    displacement2 = np.cross(a2, b)
    displacement2 = displacement2 / np.linalg.norm(displacement2)
    displacement = (la2 * displacement1 + la1 * displacement2) / (la1 + la2)
    displacement = displacement / np.linalg.norm(displacement)
    displacement = displacement * (meas[19] + offset_marker)

    pinkie_joint3_mapped = pinkie_joint3 - displacement

    xs[40] = pinkie_joint3_mapped[0]
    ys[40] = pinkie_joint3_mapped[1]
    zs[40] = pinkie_joint3_mapped[2]

    # pinkie tip
    a = pinkie_tip - pinkie_joint3
    a = a / np.linalg.norm(a)
    b = ring_mcp - pinkie_mcp
    b = b / np.linalg.norm(b)
    displacement = np.cross(a, b)
    displacement = displacement / np.linalg.norm(displacement)
    displacement = displacement * (meas[20] + offset_marker)

    pinkie_tip_mapped = pinkie_tip - displacement

    xs[41] = pinkie_tip_mapped[0]
    ys[41] = pinkie_tip_mapped[1]
    zs[41] = pinkie_tip_mapped[2]

    markers = []

    for i in range(21):
        markers.append(np.asarray([xs[21 + i], ys[21 + i], zs[21 + i]]))

    output.append(markers)

print('saving frames')

for i, f in enumerate(output):
    temp = []

    for m in f:
        temp.append(m[0])
        temp.append(m[1])
        temp.append(m[2])

    output[i] = temp

with open("mapped.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(output)
