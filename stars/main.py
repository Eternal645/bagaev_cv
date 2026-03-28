import numpy as np
from skimage.measure import label, regionprops

image_data = np.load("stars.npy")
labeled_image = label(image_data)

count_plus = 0
count_cross = 0

for region in regionprops(labeled_image):
    if region.area == 9:
        if region.image.shape[0] == 5:
            if region.image[0, 0]: 
                count_cross += 1
            else:
                count_plus += 1

print(f"Количество плюсов: {count_plus}")
print(f"Количество крестов: {count_cross}")
print(f"Количество звездочек: {count_plus + count_cross}")
