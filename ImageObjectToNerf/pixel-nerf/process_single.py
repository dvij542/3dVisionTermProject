import cv2
import numpy as np

car = cv2.imread('scene1.png')
col = car[0, 0]
mask = cv2.inRange(car, col- 2, col + 2).astype(bool)
car[mask == True] = 255
h, w = car.shape[:2]

col = car[500, 500]
mask = cv2.inRange(car, col- 2, col + 2).astype(bool)
car[mask == True] = 255
h, w = car.shape[:2]
# make it a square image through padding

if w > h:
    pad = (w - h) // 2
    car = cv2.copyMakeBorder(car, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
elif h > w:
    pad = (h - w) // 2
    car = cv2.copyMakeBorder(car, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])

print(car.shape)
# car = cv2.resize(car, (256, 256))


cv2.imwrite(f'scene1_norm.png', car)
