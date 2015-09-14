import numpy as np
import matplotlib.pyplot as plt

xs = np.arange(45, step=5)

series1 = np.array([0.0, 0.00880, 0.02660, 0.05063, 0.07714, 0.10182, 0.12235, 0.14242, 0.16158]).astype(np.double)
s1mask = np.isfinite(series1)

series2 = np.array([0.01010, 0.02948, 0.04417, 0.05932, 0.07474, 0.09013, 0.10427, 0.12280, 0.12184]).astype(np.double)
s2mask = np.isfinite(series2)

series3 = np.array([0.105893, 0.105893, 0.105893, 0.105893, 0.105893, 0.105893, 0.105893, 0.105893, 0.105893, ]).astype(np.double)
s3mask = np.isfinite(series2)

x, = plt.plot(xs[s1mask], series1[s1mask], linestyle='-', marker='o', label='item-based')
y, = plt.plot(xs[s2mask], series2[s2mask], linestyle='-', marker='o', label='clique-based')
z, = plt.plot(xs[s3mask], series3[s3mask], linestyle='-', marker='o', label='popularity')

plt.xlabel('Number of rated items', fontsize=15)
plt.ylabel('mAP@100', fontsize=15)

plt.legend(handles=[x, y, z], bbox_to_anchor=(0.35, 1))

plt.show()
