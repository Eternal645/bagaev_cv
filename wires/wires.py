import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import binary_opening

image=np.load("./wires3.npy")
struct=np.ones((3,1))
process=binary_opening(image, struct)

labeled_image=label(image)
labeled_process=label(process)
print(f"Original {np.max(labeled_image)}")
print(f"Processed {np.max(labeled_process)}")   

for wire_num in range(1, np.max(labeled_image)+1):
    wire=(labeled_image==wire_num)
    wire_parts=(labeled_process[wire])
    unique_parts=np.unique(wire_parts[wire_parts>0])
    num_parts=len(unique_parts)
    print(f"провод {wire_num},разделен на {num_parts} часть(и)")

plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(process)
plt.show()
