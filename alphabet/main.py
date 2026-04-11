import numpy as np
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.transform import resize

source_abc = imread('alphabet_ext.png', as_gray=True)
binary_abc = source_abc < 0.5
found_abc = sorted(regionprops(label(binary_abc)), key=lambda r: r.bbox[1])
labels = ['A', 'B', '8', '0', '1', 'W', 'X', '*', '-', '/', 'P', 'D']
abc_masks = {labels[i]: item.image for i, item in enumerate(found_abc)}

main_image = imread('symbols.png')
if main_image.ndim == 3:
    work_gray = np.max(main_image[..., :3], axis=2)
else:
    work_gray = main_image

objects = regionprops(label(work_gray > 10))
results = {key: 0 for key in labels}

for obj in objects:
    if obj.area < 3: 
        continue
        
    match_name = None
    max_iou = -1
    
    for key, mask in abc_masks.items():
        scaled_obj = resize(obj.image.astype(float), mask.shape, order=0) > 0.5
        
        cross = np.logical_and(scaled_obj, mask).sum()
        total = np.logical_or(scaled_obj, mask).sum()
        iou_val = cross / total if total > 0 else 0
        
        if iou_val > max_iou:
            max_iou = iou_val
            match_name = key
            
    if match_name and max_iou > 0.15: 
        results[match_name] += 1

for key in labels:
    print(f"  '{key}': {results[key]},")