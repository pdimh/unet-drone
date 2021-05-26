import csv
import cv2
import numpy as np
import os

from PIL import Image

import utils.config as config_utils
import utils.gpu as gpu
import model.unet as unet


def gen_mask(pic, class_label, sel_classes):
    # Erase classes not in CLASSES
    idx_list = [class_label.index(interest) for interest in sel_classes]
    idx_erase = np.where(np.isin(pic, idx_list, invert=True))
    pic[idx_erase[0], idx_erase[1]] = 0

    # Create mask array
    mask = np.zeros((*pic.shape, len(sel_classes)), dtype=np.float32)
    for i in range(len(idx_list)):
        idx_mask = np.where(pic == idx_list[i])
        mask[idx_mask[0], idx_mask[1], i] = 1

    return mask


config = config_utils.get_config()
gpu.configure(config.force_cpu, config.gpu_mem_limit)

PIC_PATH = os.path.join(
    config.dataset_path, 'dataset/semantic_drone_dataset/original_images')
MASK_PATH = os.path.join(
    config.dataset_path, 'dataset/semantic_drone_dataset/label_images_semantic')

class_label = []
with open(os.path.join(config.dataset_path, 'class_dict_seg.csv'), newline='') as csvfile:
    classes = csv.reader(csvfile, delimiter=',')
    next(classes, None)
    for row in classes:
        class_label.append(row[0])

pics = [cv2.resize(np.array(Image.open(os.path.join(PIC_PATH, file))), config.resolution,
                   interpolation=cv2.INTER_LINEAR) for file in sorted(
    os.listdir(PIC_PATH)
)]
mask_pics = [cv2.resize(np.array(Image.open(os.path.join(MASK_PATH, file))), config.resolution,
                        interpolation=cv2.INTER_LINEAR) for file in sorted(
    os.listdir(MASK_PATH)
)]

mask_pics = [gen_mask(pic, class_label, config.classes) for pic in mask_pics]

model = unet.get_model(pics[0].shape, n_filters=config.n_filters, n_classes=len(
    config.classes))
model.fit(np.array(pics)/255,
          np.array(mask_pics),
          batch_size=config.batch_size,
          epochs=config.epochs)

model.save_weights(config.output_path)
