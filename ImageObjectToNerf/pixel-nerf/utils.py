import numpy as np
import math

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