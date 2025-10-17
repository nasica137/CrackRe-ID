import os
import shutil
import json
import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def set_seed(seed=12091997):
    random.seed(seed)
    np.random.seed(seed)
    try:
        cv2.setRNGSeed(seed)
    except AttributeError:
        pass

# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def ensure_clean_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

def convert_numpy_to_native(data):
    if isinstance(data, dict):
        return {key: convert_numpy_to_native(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_native(value) for value in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.integer,)):
        return int(data)
    elif isinstance(data, (np.floating,)):
        return float(data)
    else:
        return data

# ------------------------------------------------------------
# Cropping
# ------------------------------------------------------------
def crop_image_and_labels(image_path, label_path, out_images_dir, out_labels_dir, object_mapping, image_id, base_name):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Fehler beim Lesen: {image_path}")
        return

    h, w = image.shape[:2]
    print(f"Verarbeite: {image_path} | Größe (WxH): {w}x{h}")

    object_count = 0

    with open(label_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            # Mindestens class_id + 1 Punktpaar
            continue

        class_id = parts[0]
        try:
            polygon_points = np.array(list(map(float, parts[1:])), dtype=float).reshape(-1, 2)
        except Exception:
            print(f"Überspringe fehlerhafte Label-Zeile in {label_path}: {line.strip()}")
            continue

        # Normierte -> Pixelkoordinaten
        px = polygon_points.copy()
        px[:, 0] *= w
        px[:, 1] *= h
        px = px.astype(np.int32)

        # Bounding Box
        min_point = np.min(px, axis=0)
        max_point = np.max(px, axis=0)

        min_x = max(int(min_point[0]), 0)
        min_y = max(int(min_point[1]), 0)
        max_x = min(int(max_point[0]), w - 1)
        max_y = min(int(max_point[1]), h - 1)

        print(f"BBox Objekt {object_count}: Min({min_x}, {min_y}), Max({max_x}, {max_y})")

        if min_x >= max_x or min_y >= max_y:
            print(f"Ungültige BBox für Objekt {object_count} in: {image_path}")
            continue

        cropped_image = image[min_y:max_y, min_x:max_x]
        if cropped_image.size == 0:
            print(f"Leerer Crop für Objekt {object_count} in: {image_path}")
            continue

        object_id = f"{image_id}_object{object_count}"

        # Speichere Crop
        output_image_path = os.path.join(out_images_dir, f"{base_name}_object{object_count}.jpg")
        cv2.imwrite(output_image_path, cropped_image)

        # Labels: normierte Original-Punkte (wie in deinem Code)
        normalized_points = polygon_points / np.array([w, h])
        normalized_points_flat = normalized_points.flatten()

        output_label_path = os.path.join(out_labels_dir, f"{base_name}_object{object_count}.txt")
        with open(output_label_path, 'w') as output_label_file:
            output_line = f"{class_id} " + " ".join(map(str, normalized_points_flat))
            output_label_file.write(output_line + "\n")

        object_mapping[object_id] = {
            "image_id": image_id,
            "class_id": class_id,
            "bbox": [min_x, min_y, max_x, max_y],
            "label_file": output_label_path,
            "image_file": output_image_path
        }

        object_count += 1

def process_cropping_for_split(split_name, base_dir, limit_images=None):
    images_dir = os.path.join(base_dir, 'images')
    labels_dir = os.path.join(base_dir, 'labels')

    output_base = f"/workspace/runs/reidentification_results/runs/{split_name}_cropped_dataset"
    ensure_clean_dir(output_base)
    out_images_dir = os.path.join(output_base, 'images')
    out_labels_dir = os.path.join(output_base, 'labels')
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    object_mapping = {}

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
    image_files.sort()
    if limit_images is not None:
        image_files = image_files[:limit_images]

    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        label_file = base_name + '.txt'
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, label_file)

        if os.path.exists(label_path):
            crop_image_and_labels(image_path, label_path, out_images_dir, out_labels_dir, object_mapping, base_name, base_name)

    mapping_file = os.path.join(output_base, 'object_mapping.json')
    with open(mapping_file, 'w') as json_file:
        json.dump(convert_numpy_to_native(object_mapping), json_file, indent=4, sort_keys=True)

    print(f"Object mapping gespeichert unter: {mapping_file}")

# ------------------------------------------------------------
# Augmentation
# ------------------------------------------------------------
def augment_image(image_path, label_path, output_image_path, output_label_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    # Random-Parameter
    angle = random.uniform(-30, 30)
    scale = random.uniform(0.8, 1.2)
    brightness = random.uniform(0.8, 1.2)

    # Affine Matrix (Rotation + Skalierung um Bildzentrum)
    cx, cy = w / 2, h / 2
    M_rot = cv2.getRotationMatrix2D((cx, cy), angle, scale)  # 2x3

    # Bild-Transformation
    img_np = np.array(image)
    img_aug = cv2.warpAffine(
        img_np, M_rot, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )

    # Helligkeit anpassen
    img_aug_pil = Image.fromarray(img_aug)
    enhancer = ImageEnhance.Brightness(img_aug_pil)
    img_aug_pil = enhancer.enhance(brightness)

    # Augmentiertes Bild speichern
    img_aug_pil.save(output_image_path)

    # Labels transformieren
    with open(label_path, 'r') as file:
        lines = file.readlines()

    with open(output_label_path, 'w') as label_out:
        for line in lines:
            if line.strip().startswith("affine_matrix"):
                continue

            parts = line.strip().split()
            if len(parts) < 3:
                continue

            class_id = parts[0]
            try:
                points = np.array(parts[1:], dtype=float).reshape(-1, 2)
            except Exception:
                continue

            # Normiert -> Pixel
            abs_points = np.zeros_like(points)
            abs_points[:, 0] = points[:, 0] * w
            abs_points[:, 1] = points[:, 1] * h

            # Homogene Koordinaten
            ones = np.ones((abs_points.shape[0], 1))
            abs_points_h = np.hstack([abs_points, ones])

            # Transformation anwenden
            aug_points = np.dot(M_rot, abs_points_h.T).T

            # Zurück normieren und clippen
            norm_points = np.zeros_like(aug_points)
            norm_points[:, 0] = np.clip(aug_points[:, 0] / w, 0, 1)
            norm_points[:, 1] = np.clip(aug_points[:, 1] / h, 0, 1)
            norm_points = norm_points.flatten()

            label_out.write(f"{class_id} {' '.join(map(str, norm_points))}\n")

        # Affine Matrix speichern
        M_flat = M_rot.flatten()
        label_out.write(f"affine_matrix: {' '.join(map(str, M_flat))}\n")

def process_augmentation_for_split(split_name, base_dir, limit_images=None):
    images_dir = os.path.join(base_dir, 'images')
    labels_dir = os.path.join(base_dir, 'labels')

    output_base = f"/workspace/runs/reidentification_results/runs/augmented_{split_name}_dataset"
    ensure_clean_dir(output_base)
    out_images_dir = os.path.join(output_base, 'images')
    out_labels_dir = os.path.join(output_base, 'labels')
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
    image_files.sort()
    if limit_images is not None:
        image_files = image_files[:limit_images]

    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        label_file = base_name + '.txt'

        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, label_file)

        output_image_path = os.path.join(out_images_dir, image_file)
        output_label_path = os.path.join(out_labels_dir, label_file)

        if os.path.exists(label_path):
            augment_image(image_path, label_path, output_image_path, output_label_path)

    print(f"Augmentiertes Dataset gespeichert unter: {output_base}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    set_seed(42)

    base_dirs = {
        'test': '/workspace/test',
        'valid': '/workspace/valid',
        'train': '/workspace/train'
    }

    for split_name, base_dir in base_dirs.items():
        limit = 1000 if split_name == 'train' else None
        process_cropping_for_split(split_name, base_dir, limit_images=limit)
        process_augmentation_for_split(split_name, base_dir, limit_images=limit)