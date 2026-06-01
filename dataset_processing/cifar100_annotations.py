import os
import json
import shutil
from typing import Dict, List, Tuple

from torchvision.datasets import CIFAR100

dataset_root = "./datasets/cifar100"

image_root = "./datasets/cifar100/image"

test_dir = "./datasets/cifar100/test"

train_dir = "./datasets/cifar100/train"

output_dir = "./datasets/cifar100/20_20_order3_2000exp"

readme_txt = "./datasets/cifar100/README.txt"

ORIGINAL_SIZE = (32, 32)

CLEAR_OLD_EXPORT = True

AUTO_CREATE_README = True

REWRITE_README = False

initial = 10
increment = 10
class_num = 100
tasks = 10

CLASS_ORDER = [
    81, 14, 3, 94, 92, 36, 49, 77, 20, 32,
    21, 47, 56, 16, 71, 27, 70, 91, 57, 26,
    74, 90, 79, 97, 23, 73, 87, 66, 84, 30,
    24, 88, 19, 93, 29, 65, 35, 4, 86, 58,
    68, 6, 0, 40, 13, 25, 75, 64, 46, 80,
    28, 83, 7, 9, 12, 37, 17, 60, 31, 33,
    8, 2, 63, 10, 18, 55, 95, 78, 85, 62,
    96, 44, 99, 76, 11, 54, 45, 5, 89, 98,
    61, 67, 69, 82, 41, 39, 42, 53, 51, 52,
    50, 1, 34, 22, 38, 15, 43, 59, 72, 48
]

IMG_EXT = ".png"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clear_dir(path: str):
    if os.path.exists(path):
        print(f"[CLEAR] remove old directory: {path}")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def move_root_files_into_cifar_python(root: str) -> str:
    cifar_py = os.path.join(root, "cifar-100-python")
    ensure_dir(cifar_py)

    for name in ["train", "test", "meta"]:
        src = os.path.join(root, name)
        dst = os.path.join(cifar_py, name)

        if os.path.exists(dst):
            continue

        if os.path.isfile(src):
            print(f"[MOVE] {src} -> {dst}")
            shutil.move(src, dst)

    return cifar_py


def create_readme_txt(classes: List[str], readme_path: str, rewrite: bool = False):
    ensure_dir(os.path.dirname(readme_path))

    if os.path.exists(readme_path) and not rewrite:
        print(f"[README] README.txt already exists, not overwritten: {readme_path}")
        return

    with open(readme_path, "w", encoding="utf-8") as f:
        for cls in classes:
            readable = cls.replace("_", " ")
            f.write(f"{cls} {readable}\n")

    print(f"[README] README.txt created: {readme_path}")


def export_train_flat(
    ds_train: CIFAR100,
    classes: List[str],
    out_root: str
) -> Tuple[int, int]:
    ensure_dir(out_root)
    saved, skipped = 0, 0

    print(f"[EXPORT] TRAIN -> {out_root} flat format, keep original 32x32")

    for idx in range(len(ds_train)):
        img, y = ds_train[idx]
        img = img.convert("RGB")

        cls = classes[int(y)]
        fname = f"{cls}_{idx:05d}{IMG_EXT}"
        fpath = os.path.join(out_root, fname)

        if os.path.exists(fpath):
            skipped += 1
            continue

        img.save(fpath)
        saved += 1

        if saved % 5000 == 0:
            print(f"  saved {saved}/{len(ds_train)} ...")

    return saved, skipped


def export_test_by_class(
    ds_test: CIFAR100,
    classes: List[str],
    out_root: str
) -> Tuple[int, int]:

    ensure_dir(out_root)

    for c in classes:
        ensure_dir(os.path.join(out_root, c))

    saved, skipped = 0, 0

    print(f"[EXPORT] TEST -> {out_root}/<class>/, keep original 32x32")

    for idx in range(len(ds_test)):
        img, y = ds_test[idx]
        img = img.convert("RGB")

        cls = classes[int(y)]
        fname = f"{cls}_{idx:05d}{IMG_EXT}"
        fpath = os.path.join(out_root, cls, fname)

        if os.path.exists(fpath):
            skipped += 1
            continue

        img.save(fpath)
        saved += 1

        if saved % 2000 == 0:
            print(f"  saved {saved}/{len(ds_test)} ...")

    return saved, skipped


def export_train_by_class(
    ds_train: CIFAR100,
    classes: List[str],
    out_root: str,
    max_images_per_class: int = 500
) -> Tuple[int, int]:
    ensure_dir(out_root)

    for c in classes:
        ensure_dir(os.path.join(out_root, c))

    saved, skipped = 0, 0
    class_count = {c: 0 for c in classes}

    print(
        f"[EXPORT] TRAIN -> {out_root}/<class>/, "
        f"max_images_per_class={max_images_per_class}, keep original 32x32"
    )

    for idx in range(len(ds_train)):
        img, y = ds_train[idx]
        img = img.convert("RGB")

        cls = classes[int(y)]

        if class_count[cls] >= max_images_per_class:
            continue

        class_dir = os.path.join(out_root, cls)
        fname = f"{cls}_{idx:05d}{IMG_EXT}"
        fpath = os.path.join(class_dir, fname)

        if os.path.exists(fpath):
            skipped += 1
            class_count[cls] += 1
            continue

        img.save(fpath)
        saved += 1
        class_count[cls] += 1

        if saved % 2000 == 0:
            print(f"  saved {saved}/{len(ds_train)} ...")

    return saved, skipped


def index_images_by_class(flat_dir: str) -> Dict[str, List[str]]:
    idx_map: Dict[str, List[str]] = {}

    if not os.path.exists(flat_dir):
        raise FileNotFoundError(f"flat_dir does not exist: {flat_dir}")

    files = [f for f in os.listdir(flat_dir) if f.lower().endswith(IMG_EXT)]
    files.sort()

    for f in files:
        stem = f[:-len(IMG_EXT)]

        if "_" not in stem:
            continue

        cls = stem.rsplit("_", 1)[0]
        idx_map.setdefault(cls, []).append(f)

    return idx_map


def build_train_task_jsons_from_image_root(
    image_root_dir: str,
    out_dir: str,
    classes: List[str]
):
    assert len(CLASS_ORDER) == 100 and sorted(CLASS_ORDER) == list(range(100)), \
        "CLASS_ORDER must be a complete permutation of 0~99"

    ensure_dir(out_dir)

    ordered_classes = [classes[i] for i in CLASS_ORDER]
    img_index = index_images_by_class(image_root_dir)

    for task_id in range(tasks):
        if task_id == 0:
            class_subset = ordered_classes[:initial]
        else:
            start = initial + (task_id - 1) * increment
            end = min(start + increment, class_num)
            class_subset = ordered_classes[start:end]

        annotations = []

        print(f"[Task {task_id}] num_classes={len(class_subset)}")

        for cls in class_subset:
            imgs = img_index.get(cls, [])

            if len(imgs) == 0:
                print(
                    f"[WARN] class={cls}: no images found in {image_root_dir}. "
                    f"Check whether TRAIN was exported correctly."
                )
                continue

            readable = cls.replace("_", " ")

            for fname in imgs:
                annotations.append({
                    "image_id": fname,
                    "caption": f"this is a photo of a {readable}.",
                })

        json_path = os.path.join(out_dir, f"task{task_id}.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"annotations": annotations}, f, indent=4, ensure_ascii=False)


def check_one_image_size(path: str):
    try:
        from PIL import Image

        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for f in files:
                    if f.lower().endswith(IMG_EXT):
                        img_path = os.path.join(root, f)
                        img = Image.open(img_path)
                        print(f"[CHECK] example image: {img_path}")
                        print(f"[CHECK] image size: {img.size}  # PIL format: (W, H)")

                        if img.size == ORIGINAL_SIZE:
                            print("[OK] CIFAR-100 image is kept as original 32x32.")
                        else:
                            print(
                                f"[WARN] image size is {img.size}, "
                                f"expected original {ORIGINAL_SIZE}."
                            )

                        img.close()
                        return

        print(f"[CHECK] No {IMG_EXT} images found to check at: {path}")

    except Exception as e:
        print(f"[CHECK] Image size check failed: {e}")


if __name__ == "__main__":
    move_root_files_into_cifar_python(dataset_root)

    if CLEAR_OLD_EXPORT:
        clear_dir(image_root)
        clear_dir(test_dir)
        clear_dir(train_dir)
        clear_dir(output_dir)
    else:
        ensure_dir(image_root)
        ensure_dir(test_dir)
        ensure_dir(train_dir)
        ensure_dir(output_dir)

    ds_train = CIFAR100(
        root=dataset_root,
        train=True,
        download=True,
        transform=None
    )

    ds_test = CIFAR100(
        root=dataset_root,
        train=False,
        download=True,
        transform=None
    )

    classes = list(ds_train.classes)

    print(f"[INFO] train samples = {len(ds_train)}")
    print(f"[INFO] test samples  = {len(ds_test)}")
    print(f"[INFO] num classes   = {len(classes)}")

    if AUTO_CREATE_README:
        create_readme_txt(classes, readme_txt, rewrite=REWRITE_README)

    saved, skipped = export_train_flat(ds_train, classes, image_root)

    saved, skipped = export_test_by_class(ds_test, classes, test_dir)

    saved, skipped = export_train_by_class(
        ds_train=ds_train,
        classes=classes,
        out_root=train_dir,
        max_images_per_class=500
    )

    build_train_task_jsons_from_image_root(
        image_root_dir=image_root,
        out_dir=output_dir,
        classes=classes
    )
    check_one_image_size(image_root)

