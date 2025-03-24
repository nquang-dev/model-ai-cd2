import cv2
import numpy as np
import os
import random

output_dirs = [
    'dataset/train/other',
    'dataset/valid/other',
    'dataset/test/other'
]

counts = [227, 49, 48]

def generate_random_images(output_dir, count):
    for i in range(count):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        for _ in range(random.randint(1, 5)):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            shape_type = random.randint(0, 2)
            pt1 = (random.randint(0, 224), random.randint(0, 224))
            pt2 = (random.randint(0, 224), random.randint(0, 224))
            
            if shape_type == 0:
                cv2.rectangle(img, pt1, pt2, color, -1)
            elif shape_type == 1:
                radius = random.randint(5, 50)
                center = (random.randint(radius, 224-radius), random.randint(radius, 224-radius))
                cv2.circle(img, center, radius, color, -1)
            else:
                thickness = random.randint(1, 10)
                cv2.line(img, pt1, pt2, color, thickness)
        
        filename = os.path.join(output_dir, f'negative_{i}.jpg')
        cv2.imwrite(filename, img)
    
    print(f"Đã tạo {count} ảnh negative trong {output_dir}")

for output_dir, count in zip(output_dirs, counts):
    generate_random_images(output_dir, count)

print("Hoàn thành tạo dữ liệu negative!")
