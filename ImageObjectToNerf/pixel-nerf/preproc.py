import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation
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
    translation = np.array(trans)
    rpy = np.array(rpy)
    rot = R.from_euler('xyz', rpy, degrees=True)
    mat = rot.as_matrix() 
    # pose = np.hstack((mat, translation))
    # pose = np.vstack((pose, np.array([0, 0, 0, 1])))
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = mat
    transformation_matrix[:3, 3] = translation
    return transformation_matrix
      
world2Camera = getMatFromPose(cameraPose[3:], cameraPose[:3])
print(world2Camera)


def TranslationMatrix(tx, ty, tz):
    """
    Returns a 4x4 translation matrix.

    :param tx: Translation in x-direction.
    :param ty: Translation in y-direction.
    :param tz: Translation in z-direction.
    :return: 4x4 translation matrix.
    """
    return np.array([[1, 0, 0, tx],
                     [0, 1, 0, ty],
                     [0, 0, 1, tz],
                     [0, 0, 0, 1]])

def RotationMatrix(roll, pitch, yaw):
    """
    Returns a 4x4 rotation matrix for a given set of Euler angles.

    :param roll: Rotation around x-axis (in radians).
    :param pitch: Rotation around y-axis (in radians).
    :param yaw: Rotation around z-axis (in radians).
    :return: 4x4 rotation matrix.
    """
    r = Rotation.from_euler('zxy', [roll, pitch, yaw], degrees=True)
    return np.vstack([np.hstack([r.as_matrix(), np.zeros((3, 1))]),
                      np.array([0, 0, 0, 1])])
focal_length_x = 1000
focal_length_y = 1000
image_center_x = im.shape[1] / 2
image_center_y = im.shape[0] / 2



rot_psi = lambda phi: np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]])

rot_theta = lambda th: np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]])

rot_phi = lambda psi: np.array([
        [np.cos(psi), -np.sin(psi), 0, 0],
        [np.sin(psi), np.cos(psi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
for i in range(2, len(data)):
    bbox = data[i][1:5]
    bbox = [int(x) for x in bbox]
    xmin, ymin, w, h = bbox
    car = im[ymin:ymin+h, xmin:xmin+w, :]

    



    col = car[0, 0]
    mask = cv2.inRange(car, col- 2, col + 2).astype(bool)
    car[mask == True] = 255

    lab_img = cv2.cvtColor(car, cv2.COLOR_BGR2LAB)

    # Split the LAB image into its 3 channels
    l, a, b = cv2.split(lab_img)

    # Apply histogram equalization to the L channel
    # l_eq = cv2.equalizeHist(l)
    l_eq = clahe.apply(l)


    lab_eq = cv2.merge((l_eq, a, b))

# Convert the equalized LAB image back to BGR color space
    car = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # make it a square image through padding

    if w > h:
        pad = (w - h) // 2
        car = cv2.copyMakeBorder(car, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    elif h > w:
        pad = (h - w) // 2
        car = cv2.copyMakeBorder(car, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    padding = 80
    car = cv2.copyMakeBorder(car, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    print(car.shape)
    car = cv2.resize(car, (256, 256))


    cv2.imwrite(f'my_input/rgb/{data[i][0]}.png', car)

    objWorldPose = data[i][8:]
    objWorldPose = [float(x) for x in objWorldPose]
    objWorldPose = np.array(objWorldPose)
    # objWorldPose[3] = 90 - objWorldPose[3]
    elevation = - (cameraPose[3] - objWorldPose[3])
    azimuth = - (cameraPose[4] - objWorldPose[4])
    yaw = - (cameraPose[5] - objWorldPose[5])

    roll = objWorldPose[3]
    pitch = objWorldPose[4]
    yaw = objWorldPose[5]


    # world2Camera = getMatFromPose(cameraPose[3:], cameraPose[:3])
    # camera2World = np.linalg.inv(world2Camera)
    # object2World = getMatFromPose(objWorldPose[3:], objWorldPose[:3])
    # object2Camera = np.matmul(camera2World, object2World)

    # camera2Object = np.linalg.inv(object2Camera)    
    # object2Camera = getMatFromPose([68.4, 0, 11.5], [0, 0, 0])
    object2Camera = R.from_quat((0.559, 0.056, 0.083, 0.823))
    # object2Camera = object2Camera.as_euler('xyz', degrees=True)
    print(object2Camera.as_euler('xyz', degrees=True))

    object2Camera = R.from_euler('xyz', [90, 0, 0], degrees=True)
    object2Camera = np.vstack([np.hstack([object2Camera.as_matrix(), np.zeros((3, 1))]), np.array([0, 0, 0, 1])])
    mat = object2Camera.copy()
    mat[:3, 3] = 0
    # mat[2, 3] = 0.5
    # camera2Object[:3, 3] = 0
    np.save(f'my_input/pose/{data[i][0]}.npy', mat)
    # np.save(f'my_input/pose/car7.npy', M_camera_to_object_crop)







