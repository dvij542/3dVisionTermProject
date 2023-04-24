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
    translation = trans
    rot = R.from_euler('xyz', rpy, degrees=True)
    mat = rot.as_matrix() 
    # pose = np.hstack((mat, translation))
    # pose = np.vstack((pose, np.array([0, 0, 0, 1])))
    transformation_matrix = np.zeros((3, 4))
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

for i in range(2, len(data)):
    bbox = data[i][1:5]
    bbox = [int(x) for x in bbox]
    xmin, ymin, w, h = bbox
    car = im[ymin:ymin+h, xmin:xmin+w, :]
    col = car[0, 0]
    mask = cv2.inRange(car, col- 2, col + 2).astype(bool)
    car[mask == True] = 255

    # make it a square image through padding

    if w > h:
        pad = (w - h) // 2
        car = cv2.copyMakeBorder(car, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    elif h > w:
        pad = (h - w) // 2
        car = cv2.copyMakeBorder(car, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    print(car.shape)
    car = cv2.resize(car, (256, 256))


    cv2.imwrite(f'my_input/rgb/{data[i][0]}.png', car)

    objWorldPose = data[i][8:]
    objWorldPose = [float(x) for x in objWorldPose]
    objWorldPose = np.array(objWorldPose)
    # objWorldPose[3] = 90 - objWorldPose[3]
    
    # azimuth = -(cameraPose[4] - objWorldPose[4])
    # yaw = -(cameraPose[5] - objWorldPose[5])
    # objWorldPose[5] = 180 - objWorldPose[5]
    # obj

    #     # Calculate the camera intrinsics matrix, which defines the relationship between image pixels and real-world measurements
    # K = np.array([[focal_length_x, 0, image_center_x],
    #             [0, focal_length_y, image_center_y],
    #             [0, 0, 1]])
    
    # # Calculate the relative transformation from the camera to the object after cropping
    # x_offset = xmin + W/2 - image_center_x
    # y_offset = ymin + H/2 - image_center_y
    # depth_scale = np.sqrt(W**2 + H**2) / np.sqrt(w**2 + h**2)

    # T_crop = np.array([[1, 0, 0, -x_offset / focal_length_x],
    #                 [0, 1, 0, -y_offset / focal_length_y],
    #                 [0, 0, 1, depth_scale],
    #                 [0, 0, 0, 1]])  # Translation and scaling matrix for cropping

    # R_camera = RotationMatrix(*cameraPose[3:])  # Rotation matrix for camera
    # T_camera = TranslationMatrix(*cameraPose[:3])  # Translation matrix for camera

    # R_object = RotationMatrix(*objWorldPose[3:])  # Rotation matrix for object
    # T_object = TranslationMatrix(*objWorldPose[:3])  # Translation matrix for object

    # # Compose the transformation matrices
    # M_camera = T_camera @ R_camera
    # M_object = T_object @ R_object
    # M_crop = T_crop
    # # print(M_camera.shape,   M_object.shape, M_crop.shape)
    # M_camera_to_object_crop = M_object @ np.linalg.inv(M_crop) @ np.linalg.inv(M_camera)  # Relative transformation from camera to object after cropping
    # M_camera_to_object_crop[:3, 3] = np.zeros((3,))
    M_camera_to_object_crop = np.eye(4)
    arr = [[
                    -0.6977230310440063,
                    0.03695093095302582,
                    0.7154139280319214,
                    1.4308279752731323
                ],
                [
                    0.7163675427436829,
                    0.03598923236131668,
                    0.6967942118644714,
                    1.3935885429382324
                ],
                [
                    -3.725290076417309e-09,
                    0.9986687898635864,
                    -0.051580965518951416,
                    -0.10316193848848343
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ]
    # M_camera_to_object_crop = np.array(arr)
    # r, p, y = R.from_matrix(M_camera_to_object_crop[:3, :3]).as_euler('xyz', degrees=True)
    # print(r, p, y)
    # r = rot_psi(np.deg2rad(10))
    r = rot_theta(np.deg2rad(45))
    # r = rot_phi(np.deg2rad(90))
    # phi, theta, psi, t = kwargs
    # start_pose = rot_phi(phi/180.*np.pi) @ rot_theta(theta/180.*np.pi) @ rot_psi(psi/180.*np.pi)  @ obs_img_pose
    # start_pose = trans_t(t[i]) @ np.linalg.inv(start_pose)
    # start_pose = np.linalg.inv(start_pose)
    M_camera_to_object_crop = r
    # M_camera_to_object_crop[0, -1] = 0.5
    # M_camera_to_object_crop[1, -1] = 0.5
    M_camera_to_object_crop = np.linalg.inv(M_camera_to_object_crop)
    M_camera_to_object_crop[2, -1] = 1.3
    # print(roll, pitch, yaw)
    # np.save(f'my_input/pose/{data[i][0]}.npy', M_camera_to_object_crop)
    np.save(f'my_input/pose/car7.npy', M_camera_to_object_crop)







