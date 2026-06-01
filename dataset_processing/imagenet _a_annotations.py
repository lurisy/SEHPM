import os
import json
import shutil
from typing import Dict, List, Tuple
from torchvision.datasets import ImageFolder

import os
import shutil

train_root = "./datasets/imagenet-a/train"
image_root = "./datasets/imagenet-a/image"

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

output_dir = "./datasets/imagenet-a/20_20_order3_2000exp"

initial = 10
increment = 10
class_num = 200
tasks = 20

class_names = sorted([
    d for d in os.listdir(train_root)
    if os.path.isdir(os.path.join(train_root, d))
])

wnid_to_word = {}
words_txt = "./datasets/imagenet-a/README.txt"

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



class_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]

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
            print(f"[WARN] No images found for class {cls_name} in image/. Check the data split script.")
            continue


        for img in cls_images:
            annotations.append({
                "image_id": img,
                "caption": f"this is a photo of a {readable_name}.",
            })

    json_path = os.path.join(output_dir, f"task{task_id}.json")
    with open(json_path, "w") as f:
        json.dump({"annotations": annotations}, f, indent=4)

    print(f"[Task {task_id}] Saved to {json_path}, with {len(annotations)} samples in total.")
