import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation
import math
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

trans_t = lambda t: np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]])


def getObjectPose(bbox,opt_pose,camera_angle,dist,W, H) :
    """
    Inputs :-
        bbox : (x,y,w,h) bounding box
        opt_pose : camera local -> object transformation
        camera_angle : Field of view of the camera
        W : Width of the scene image
        H : Height of the scene image
    Returns :-
        world -> object transformation
        obj_dist : Object distance from local camera pose
    """
    # local_dist = 2.
    # print(camera_angle)
    # object_size = 2.*local_dist*math.tan(camera_angle/2.)
    # print(W,bbox[2])
    # dist = object_size*W/bbox[2]
    # print(dist)
    x,y,w,h = bbox
    world_to_cam = rot_psi(-21.6*math.pi/180.)@np.array([[1.,0.,0.,0.],\
                             [0.,1.,0.,0.],\
                             [0.,0.,1.,8.1],\
                             [0.,0.,0.,1.]])
    cx = W/2 - (x+w/2)
    cy = (y+h/2) - H/2
    D = (W/2)/(math.tan(camera_angle/2))
    # print("Dip : ", math.atan(cy/D)*180./math.pi,cx)
    obj_to_caml = rot_theta(math.pi) @ np.linalg.inv(opt_pose)
    obj_dist = np.linalg.norm(obj_to_caml[:3,-1])

    caml_to_cam = rot_theta(math.atan(-cx/D))@rot_psi(math.atan(cy/D))@\
                   np.array([[1.,0.,0.,0.],\
                             [0.,1.,0.,0.],\
                             [0.,0.,1.,dist-obj_dist],\
                             [0.,0.,0.,1.]])
    
    object_to_world = np.linalg.inv(world_to_cam) @ caml_to_cam @ obj_to_caml
    print("Object -> world transformation : ",object_to_world)
    # x,y,z,r,p,yaw = opt_pose
    # obj_angle = np.array([r,p,yaw])
    return object_to_world, obj_dist

def getObjectRelPose(object_pose,world_to_camera,camera_angle,obj_dist,W,H) :
    """
    Inputs :-
        object_pose : object -> world transformation
        world_to_camera : world -> camera transformation
        camera_angle : Field of view of the camera
        obj_dist : Object distance from camera
        W : Width of the scene image
        H : Height of the scene image
    Returns :-
        camera local -> object transformation
    """
    object_to_camera = world_to_camera @ object_pose
    dist = np.linalg.norm(object_to_camera[:3,3])
    object_to_camlocal = rot_theta(math.pi) @ np.array([[1.,0.,0.,0.],\
                             [0.,1.,0.,0.],\
                             [0.,0.,1.,-dist+obj_dist],\
                             [0.,0.,0.,1.]])@rot_psi(math.asin(object_to_camera[1,3]/dist))@rot_theta(math.asin(object_to_camera[0,3]/dist))@object_to_camera
    # print(rot_psi(math.asin(object_to_camera[1,3]/dist))@rot_theta(math.asin(object_to_camera[0,3]/dist))@object_to_camera)
    D = (W/2)/(math.tan(camera_angle/2))
    # print(math.asin(object_to_camera[0,3]/dist),camera_angle/2,W)
    # print(math.asin(object_to_camera[1,3]/dist),camera_angle/2,H)
    y = int(W/2 - D*math.tan(math.asin(object_to_camera[0,3]/dist)))
    x = int(H/2 - D*math.tan(math.asin(object_to_camera[1,3]/dist)))
    return np.linalg.inv(object_to_camlocal), x, y
focal_length_x = 1000
focal_length_y = 1000
image_center_x = im.shape[1] / 2
image_center_y = im.shape[0] / 2

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

        # Calculate the camera intrinsics matrix, which defines the relationship between image pixels and real-world measurements
    
    np.save(f'my_input/pose/{data[i][0]}.npy', M_camera_to_object_crop)







