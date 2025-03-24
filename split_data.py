import os
import random
import shutil
from pathlib import Path

source_dir = '/home/nquang/D_DAI-HOC/CD2/traffic_sign_project/3'
output_dir = 'dataset'

for split in ['train', 'valid', 'test']:
    for category in ['40kmh', 'other']:
        os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)

all_images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(all_images)

total = len(all_images)
train_count = int(0.7 * total)
valid_count = int(0.15 * total)
test_count = total - train_count - valid_count

print(f"Tổng số ảnh: {total}")
print(f"Chia thành: Train={train_count}, Validation={valid_count}, Test={test_count}")

train_images = all_images[:train_count]
valid_images = all_images[train_count:train_count+valid_count]
test_images = all_images[train_count+valid_count:]

def copy_images(image_list, source, destination):
    for img in image_list:
        src_path = os.path.join(source, img)
        dst_path = os.path.join(destination, img)
        shutil.copy2(src_path, dst_path)
    print(f"Đã copy {len(image_list)} ảnh vào {destination}")

copy_images(train_images, source_dir, os.path.join(output_dir, 'train', '40kmh'))
copy_images(valid_images, source_dir, os.path.join(output_dir, 'valid', '40kmh'))
copy_images(test_images, source_dir, os.path.join(output_dir, 'test', '40kmh'))

print("Hoàn thành phân chia dữ liệu!")
