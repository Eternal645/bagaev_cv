import cv2
import numpy as np

img_source = cv2.imread("C:/Users/kemmi/Documents/comp_zrenie/figures_and_colors/balls_and_rects.png")
hsv_image = cv2.cvtColor(img_source, cv2.COLOR_BGR2HSV)

color_boundaries = {
    "Красный": ((0, 50, 50), (10, 255, 255)),
    "Оранжевый": ((11, 50, 50), (25, 255, 255)),
    "Жёлтый": ((26, 50, 50), (35, 255, 255)),
    "Зелёный": ((36, 50, 50), (85, 255, 255)),
    "Синий": ((86, 50, 50), (130, 255, 255)),
    "Фиолетовый": ((131, 50, 50), (160, 255, 255)),
    "Розовый": ((161, 50, 50), (180, 255, 255))
}

total_objects_found = 0

for color_title, (hsv_min, hsv_max) in color_boundaries.items():
    
    color_mask = cv2.inRange(hsv_image, np.array(hsv_min), np.array(hsv_max))
    contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count_circles = 0
    count_rectangles = 0
    
    for shape in contours:
        shape_area = cv2.contourArea(shape)
        
        if shape_area < 10: 
            continue
            
        shape_perimeter = cv2.arcLength(shape, True)
        if shape_perimeter == 0: 
            continue
        
        roundness_factor = (4 * np.pi * shape_area) / (shape_perimeter ** 2)
        
        if roundness_factor >= 0.85: 
            count_circles += 1
        else:
            count_rectangles += 1
            
        total_objects_found += 1
    
    if count_circles > 0 or count_rectangles > 0:
        print(f"Цвет: {color_title}")
        print(f"Кругов: {count_circles}")
        print(f"Прямоугольников: {count_rectangles}")

print(f"Общее число распознанных фигур: {total_objects_found}")