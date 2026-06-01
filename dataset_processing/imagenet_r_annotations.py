import os
import json
import shutil
from typing import Dict, List, Tuple
from torchvision.datasets import ImageFolder

import os
import shutil

train_root = "./datasets/imagenet-r/train"
image_root = "./datasets/imagenet-r/image"

if os.path.exists(image_root):
    shutil.rmtree(image_root)
os.makedirs(image_root, exist_ok=True)

img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".JPEG", ".JPG", ".PNG")

total = 0
for wnid in sorted(os.listdir(train_root)):
    cls_dir = os.path.join(train_root, wnid)
    if not os.path.isdir(cls_dir):
        continue

    imgs = sorted([
        f for f in os.listdir(cls_dir)
        if f.lower().endswith(tuple(e.lower() for e in img_exts))
    ])

    for idx, img in enumerate(imgs):
        src = os.path.join(cls_dir, img)
        ext = os.path.splitext(img)[1]

        dst_name = f"{wnid}_{idx:06d}{ext}"
        dst = os.path.join(image_root, dst_name)

        shutil.copy2(src, dst)
        total += 1

    print(f"[FLATTEN] {wnid}: {len(imgs)} images")

print(f"[DONE] total images copied: {total}")

output_dir = "./datasets/imagenet-r/20_20_order3_2000exp"


initial = 20
increment = 20
class_num = 200
tasks = 10

class_names = sorted([
    d for d in os.listdir(train_root)
    if os.path.isdir(os.path.join(train_root, d))
])

wnid_to_word = {}
words_txt = "./datasets/imagenet-r/README.txt"

with open(words_txt, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        wnid, words = parts
        wnid_to_word[wnid] = words.replace("_", " ")



class_order = [168, 136, 51, 9, 183, 101, 171, 99, 42, 159, 191, 70, 16, 188, 27, 10, 175, 26, 68, 187, 98, 6, 85, 35, 112, 43, 100, 0, 103, 181, 88, 59, 4, 2, 116, 174, 94, 80, 106, 1, 147, 17, 141, 131, 72, 23, 173, 54, 197, 118, 87, 32, 79, 104, 91, 19, 135, 107, 178, 36, 11, 199, 142, 8, 122, 3, 28, 57, 153, 172, 190, 56, 49, 44, 97, 62, 151, 169, 194, 55, 192, 12, 189, 78, 66, 180, 15, 137, 109, 134, 92, 119, 126, 52, 170, 40, 148, 65, 144, 64, 138, 45, 77, 89, 154, 90, 71, 193, 74, 30, 113, 143, 96, 84, 67, 50, 186, 156, 69, 21, 18, 111, 108, 58, 125, 157, 150, 110, 182, 129, 166, 83, 81, 60, 13, 165, 14, 176, 63, 117, 5, 22, 145, 121, 38, 41, 82, 127, 114, 20, 31, 53, 37, 163, 196, 130, 152, 162, 86, 76, 24, 34, 184, 149, 33, 128, 198, 155, 146, 167, 139, 120, 140, 102, 47, 25, 158, 123, 46, 164, 61, 7, 115, 75, 133, 160, 105, 132, 179, 124, 48, 73, 93, 39, 95, 195, 29, 177, 185, 161]

ordered_class_names = [class_names[i] for i in class_order]

os.makedirs(output_dir, exist_ok=True)

for task_id in range(tasks):
    annotations = []

    if task_id == 0:
        class_subset = ordered_class_names[:initial]
    else:
        task_class_start = initial + (task_id - 1) * increment
        task_class_end = task_class_start + increment
        if task_class_end > class_num:
            task_class_end = class_num
        class_subset = ordered_class_names[task_class_start:task_class_end]

    print(f"[Task {task_id}] num_classes = {len(class_subset)}")

    for cls_name in class_subset:
        readable_name = wnid_to_word.get(cls_name, cls_name)

        cls_images = sorted([
            f for f in os.listdir(image_root)
            if f.startswith(cls_name)
        ])

        if len(cls_images) == 0:
            print(f"[WARN] No images found for class {cls_name} under image/. Please check the data splitting script.")
            continue


        for img in cls_images:
            annotations.append({
                "image_id": img,
                "caption": f"this is a photo of a {readable_name}.",
            })

    json_path = os.path.join(output_dir, f"task{task_id}.json")
    with open(json_path, "w") as f:
        json.dump({"annotations": annotations}, f, indent=4)

