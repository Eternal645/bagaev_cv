import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

output_dir = Path(__file__).parent / "recognition_results"
output_dir.mkdir(exist_ok=True)

def get_holes_count(region_obj):
    padded_img = np.pad(region_obj.image, pad_width=1, mode='constant', constant_values=0)
    inverted = np.logical_not(padded_img)
    labeled_bg = label(inverted)
    return np.max(labeled_bg)

def calc_lower_density(img_matrix):
    h, w = img_matrix.shape
    dy = max(1, h // 10)
    dx = max(1, w // 10)
    
    left_corner = img_matrix[-dy:, :dx].mean()
    right_corner = img_matrix[-dy:, -dx:].mean()
    return (left_corner + right_corner) / 2.0

def extract_features(region_obj):
    img = region_obj.image
    height, width = img.shape
    
    y0, x0 = region_obj.centroid_local
    norm_y = y0 / height
    norm_x = x0 / width
    
    num_holes = get_holes_count(region_obj)
    ecc = region_obj.eccentricity
    bottom_corners = calc_lower_density(img)
    aspect_ratio = height / width
    
    return np.array([num_holes, aspect_ratio, ecc, norm_y, norm_x, bottom_corners])

def predict_symbol(target_features, reference_dict):
    best_match = "?"
    min_dist = float('inf')
    
    for sym, ref_features in reference_dict.items():
        dist = np.linalg.norm(ref_features - target_features)
        if dist < min_dist:
            min_dist = dist
            best_match = sym
            
    return best_match

template_image = imread("alphabet-small.png")[:, :, :3].sum(axis=2)
binary_template = template_image != 765.0

labeled_template = label(binary_template)
template_props = regionprops(labeled_template)

known_classes = ["8", "0", "A", "B", "1", "W", "X", "*", "/", "-"]
reference_data = {}

for prop, symbol in zip(template_props, known_classes):
    reference_data[symbol] = extract_features(prop)

target_image = imread("alphabet.png")[:, :, :3]
binary_target = target_image.mean(axis=2) > 0
labeled_target = label(binary_target)

print(f"Всего объектов найдено: {np.max(labeled_target)}")

target_props = regionprops(labeled_target)
stats = {}

plt.figure(figsize=(5, 7))

for idx, prop in enumerate(target_props):
    predicted = predict_symbol(extract_features(prop), reference_data)
    
    stats[predicted] = stats.get(predicted, 0) + 1
    
    plt.clf()
    plt.title(f"класс: '{predicted}'")
    plt.imshow(prop.image, cmap='gray')
    plt.savefig(output_dir / f"char_{prop.label:03d}.png")

print("\nстатистика распознавания:", stats)

errors = stats.get("?", 0)
accuracy = (1 - errors / len(target_props)) * 100
print(f"пРоцент распознавания: {accuracy:.2f}%")

plt.imshow(binary_target, cmap='gray')
plt.title("изображение для распознавания")
plt.show()
