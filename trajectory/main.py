import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def get_centers(idx):
    img = np.load(f'out/h_{idx}.npy')
    lbl, n = ndimage.label(img)
    return np.array(ndimage.center_of_mass(img, lbl, range(1, n + 1)))

data0 = get_centers(0)
tracks = [[p] for p in data0[data0[:, 0].argsort()]]

for i in range(1, 100):
    points = get_centers(i)
    for t in tracks:
        dists = np.hypot(*(points - t[-1]).T)
        t.append(points[np.argmin(dists)])

colors = plt.cm.magma(np.linspace(0.1, 0.8, len(tracks)))

plt.figure(figsize=(8, 5))
for i, path in enumerate(tracks):
    p = np.array(path)
    plt.plot(p[:, 1], p[:, 0], color=colors[i], lw=1.5, marker='.', ms=3)

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()