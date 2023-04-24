import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

os.makedirs('my_input/rgb', exist_ok=True)
os.makedirs('my_input/pose', exist_ok=True)
im  = cv2.imread('my_input/scene1.png')
with open('my_input/data.txt', 'r') as f:
    data = f.readlines()
    data = [x.split(',') for x in data]

keys = data[0]

H, W = im.shape[:2]
cameraPose = np.array(list(map(float, data[1][8:]))) # x, y, z, r, p, yaw
print(cameraPose)
np.save(f'my_input/pose/camerapose.npy', cameraPose)

def getMatFromPose(rpy, trans):
    translation = trans.reshape(3, 1)
    rot = R.from_euler('xyz', rpy, degrees=True)
    mat = rot.as_matrix() 
    pose = np.hstack((mat, translation))
    pose = np.vstack((pose, np.array([0, 0, 0, 1])))

    return pose
      
world2Camera = getMatFromPose(cameraPose[3:], cameraPose[:3])
print(world2Camera)


for i in range(2, len(data)):
    bbox = data[i][1:5]
    bbox = [int(x) for x in bbox]
    x1, y1, w, h = bbox
    car = im[y1:y1+h, x1:x1+w, :]
    col = car[0, 0]
    mask = cv2.inRange(car, col- 2, col + 2).astype(bool)
    car[mask == True] = 255
    cv2.imwrite(f'my_input/rgb/{data[i][0]}.png', car)

    pose = data[i][8:]
    pose = [float(x) for x in pose]
    pose = np.array(pose)

    world2Obj = getMatFromPose(pose[3:], pose[:3])
    pose = world2Camera @ world2Obj
    np.save(f'my_input/pose/{data[i][0]}.npy', pose)







