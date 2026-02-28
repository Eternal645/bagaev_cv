import numpy as np
import matplotlib.pyplot as plt

def lerp(v0, v1, t):
    return (1 - t) * v0 + t * v1

size = 100
image = np.zeros((size, size, 3), dtype="uint8")

color1 = [255, 20, 147]
color2 = [25, 25, 112] 
  
max_sum = 2 * (size - 1)

for i1 in range(size):
    for i2 in range(size):
        t = (i1 + i2) / max_sum
        
        r = lerp(color1[0], color2[0], t)
        g = lerp(color1[1], color2[1], t)
        b = lerp(color1[2], color2[2], t)
        
        image[i1, i2] = [r, g, b]

plt.figure(1)
plt.imshow(image)
plt.show()