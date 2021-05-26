import argparse
import cv2

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import utils.config as config_utils
import utils.gpu as gpu
import model.unet as unet


parser = argparse.ArgumentParser(
    description='Detect classes (drone dataset) using trained model')
parser.add_argument('path',
                    help='path of the picture')
args = parser.parse_args()

try:
    data = np.array(Image.open(args.path))
except IOError:
    print('Error: File not found.')
    exit(1)

config = config_utils.get_config()
gpu.configure(config.force_cpu, config.gpu_mem_limit)

data = cv2.resize(data, config.resolution,
                  interpolation=cv2.INTER_LINEAR)

model = unet.get_model(data.shape, n_filters=config.n_filters, n_classes=len(
    config.classes))
model.load_weights(config.output_path)

prediction = model.predict(np.array([data])/255)
prediction = np.squeeze(prediction, axis=0)

out = np.zeros(prediction.shape[0:2])

out = np.argmax(prediction, axis=2)
out = np.ma.masked_where(out == 0, out)

plt.imshow(data)
plt.imshow(out, 'jet', alpha=0.7)
plt.show()
