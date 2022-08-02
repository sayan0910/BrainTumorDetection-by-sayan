import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10Epochs.h5')

image=cv2.imread('C:\\Users\\DESKTOP\Desktop\\brain_tumor\\pred\\pred1.jpg')

img=Image.fromarray(image)

img=image.resize(64,64)

img=np.array(img)

print(img)