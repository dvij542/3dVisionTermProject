import os
import torch
import imageio
import numpy as np
import skimage
import cv2
from utils import config_parser, load_blender, load_car, show_img, find_POI, img2mse, load_llff_data, getObjectPose, getObjectRelPose
from nerf_helpers import load_nerf
from render_helpers import render, to8b, get_rays
from inerf_helpers import camera_transf
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

rot_theta = lambda th: np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]])

rot_psi = lambda phi: np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]])

def get_triple_sized(img) :
    print(img.shape)
    img_new = np.zeros((3*img.shape[0],3*img.shape[1],3))
    for i in range(3) :
        for j in range(3) :
            img_new[i::3,j::3,:] = img
            
    return img_new



def run():

    # Parameters
    parser = config_parser()
    args = parser.parse_args()
    output_dir = args.output_dir
    if 'car' in args.model_name :
        model_name = ['car1','car2']
        args.model_name = ['car1','car2']
    if args.obs_img_num==24 :
        obs_img_num = [24,15]
        args.obs_img_num = [24,15]
    batch_size = args.batch_size
    spherify = args.spherify
    kernel_size = args.kernel_size
    lrate = args.lrate
    dataset_type = args.dataset_type
    sampling_strategy = args.sampling_strategy
    delta_phi, delta_theta, delta_psi, delta_t = args.delta_phi, args.delta_theta, args.delta_psi, args.delta_t
    noise, sigma, amount = args.noise, args.sigma, args.amount
    delta_brightness = args.delta_brightness

    # Load and pre-process an observed image
    # obs_img -> rgb image with elements in range 0...255
    if dataset_type == 'blender':
        obs_img, hwf, start_pose, obs_img_pose = load_blender(args.data_dir, model_name, obs_img_num,
                                                args.half_res, args.white_bkgd, delta_phi, delta_theta, delta_psi, delta_t)
        H, W, focal = hwf
        near, far = 1., 3.  # Blender
    elif dataset_type == 'car':
        obs_img_, hwf, start_poses, obs_img_poses, bbox, camera_angle, H_, W_ = load_car(args.data_dir, model_name, obs_img_num,
                                                args.half_res, args.white_bkgd, (delta_phi, delta_theta, delta_psi, delta_t),
                                                scene_dir='../../data/scenes/scene1_without_shadows.png', 
                                                bboxes_file='../../data/bboxes/1.txt',)
        H, W, focal = hwf
        near, far = .5, 3.5  # Blender
    else:
        obs_img, hwf, start_pose, obs_img_pose, bds = load_llff_data(args.data_dir, model_name, obs_img_num, delta_phi,
                                                delta_theta, delta_psi, delta_t, factor=8, recenter=True, bd_factor=.75, spherify=spherify)
        H, W, focal = hwf
        H, W = int(H), int(W)
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.

    # exit(0)
    # Load NeRF Model
    render_kwargs_ = []
    object_poses = []
    object_dists = []
    for i in range(2) :
        # print(i)
        obs_img_pose = obs_img_poses[i]
        start_pose = start_poses[i]
        obs_img = (np.array(obs_img_[i]) / 255.).astype(np.float32)
        # print(np.min(obs_img),np.max(obs_img))
        # print(obs_img.shape)
        # change brightness of the observed image (to test robustness of inerf)
        if delta_brightness != 0:
            obs_img = (np.array(obs_img) / 255.).astype(np.float32)
            obs_img = cv2.cvtColor(obs_img, cv2.COLOR_RGB2HSV)
            if delta_brightness < 0:
                obs_img[..., 2][obs_img[..., 2] < abs(delta_brightness)] = 0.
                obs_img[..., 2][obs_img[..., 2] >= abs(delta_brightness)] += delta_brightness
            else:
                lim = 1. - delta_brightness
                obs_img[..., 2][obs_img[..., 2] > lim] = 1.
                obs_img[..., 2][obs_img[..., 2] <= lim] += delta_brightness
            obs_img = cv2.cvtColor(obs_img, cv2.COLOR_HSV2RGB)
            show_img("Observed image", obs_img)
        
        # apply noise to the observed image (to test robustness of inerf)
        if noise == 'gaussian':
            obs_img_noised = skimage.util.random_noise(obs_img, mode='gaussian', var=sigma**2)
        elif noise == 's_and_p':
            obs_img_noised = skimage.util.random_noise(obs_img, mode='s&p', amount=amount)
        elif noise == 'pepper':
            obs_img_noised = skimage.util.random_noise(obs_img, mode='pepper', amount=amount)
        elif noise == 'salt':
            obs_img_noised = skimage.util.random_noise(obs_img, mode='salt', amount=amount)
        elif noise == 'poisson':
            obs_img_noised = skimage.util.random_noise(obs_img, mode='poisson')
        else:
            obs_img_noised = obs_img

        obs_img_noised = (np.array(obs_img_noised) * 255).astype(np.uint8)
        if DEBUG:
            show_img("Observed image", obs_img_noised)


        # find points of interest of the observed image
        POI = find_POI(obs_img_noised, DEBUG)  # xy pixel coordinates of points of interest (N x 2)
        # print(POI.shape)
        obs_img_noised = (np.array(obs_img_noised) / 255.).astype(np.float32)

        # create meshgrid from the observed image
        coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, W[i] - 1, W[i]), np.linspace(0, H[i] - 1, H[i])), -1),
                            dtype=int)

        # create sampling mask for interest region sampling strategy
        interest_regions = np.zeros((H[i], W[i], ), dtype=np.uint8)
        interest_regions[POI[:,1], POI[:,0]] = 1
        I = args.dil_iter
        interest_regions = cv2.dilate(interest_regions, np.ones((kernel_size, kernel_size), np.uint8), iterations=I)
        interest_regions = np.array(interest_regions, dtype=bool)
        interest_regions = coords[interest_regions]

        # not_POI -> contains all points except of POI
        coords = coords.reshape(H[i] * W[i], 2)
        not_POI = set(tuple(point) for point in coords) - set(tuple(point) for point in POI)
        not_POI = np.array([list(point) for point in not_POI]).astype(int)

        args.model_name = model_name[i]
        render_kwargs = load_nerf(args, device)
        bds_dict = {
            'near': near,
            'far': far,
        }
        render_kwargs.update(bds_dict)
        render_kwargs_.append(render_kwargs)

        # Create pose transformation model
        start_pose = torch.Tensor(start_pose).to(device)
        cam_transf = camera_transf().to(device)
        optimizer = torch.optim.Adam(params=cam_transf.parameters(), lr=lrate, betas=(0.9, 0.999))

        # calculate angles and translation of the observed image's pose
        phi_ref = np.arctan2(obs_img_pose[1,0], obs_img_pose[0,0])*180/np.pi
        theta_ref = np.arctan2(-obs_img_pose[2, 0], np.sqrt(obs_img_pose[2, 1]**2 + obs_img_pose[2, 2]**2))*180/np.pi
        psi_ref = np.arctan2(obs_img_pose[2, 1], obs_img_pose[2, 2])*180/np.pi
        translation_ref = np.sqrt(obs_img_pose[0,3]**2 + obs_img_pose[1,3]**2 + obs_img_pose[2,3]**2)
        #translation_ref = obs_img_pose[2, 3]

        testsavedir = os.path.join(output_dir, model_name[i])
        os.makedirs(testsavedir, exist_ok=True)

        # imgs - array with images are used to create a video of optimization process
        if OVERLAY is True:
            imgs = []

        for k in range(600):

            if sampling_strategy == 'random':
                rand_inds = np.random.choice(coords.shape[0], size=batch_size, replace=False)
                batch = coords[rand_inds]

            elif sampling_strategy == 'interest_points':
                if POI.shape[0] >= batch_size:
                    rand_inds = np.random.choice(POI.shape[0], size=batch_size, replace=False)
                    batch = POI[rand_inds]
                else:
                    batch = np.zeros((batch_size, 2), dtype=np.int)
                    batch[:POI.shape[0]] = POI
                    rand_inds = np.random.choice(not_POI.shape[0], size=batch_size-POI.shape[0], replace=False)
                    batch[POI.shape[0]:] = not_POI[rand_inds]

            elif sampling_strategy == 'interest_regions':
                rand_inds = np.random.choice(interest_regions.shape[0], size=batch_size, replace=False)
                batch = interest_regions[rand_inds]

            else:
                print('Unknown sampling strategy')
                return

            target_s = obs_img_noised[batch[:, 1], batch[:, 0]]
            target_s = torch.Tensor(target_s).to(device)
            # print("Start pose : ", start_pose)
            pose = cam_transf(start_pose)

            rays_o, rays_d = get_rays(H[i], W[i], focal[i], pose)  # (H, W, 3), (H, W, 3)
            rays_o = rays_o[batch[:, 1], batch[:, 0]]  # (N_rand, 3)
            rays_d = rays_d[batch[:, 1], batch[:, 0]]
            batch_rays = torch.stack([rays_o, rays_d], 0)

            rgb, disp, acc, extras = render(H[i], W[i], focal[i], chunk=args.chunk, rays=batch_rays,
                                            verbose=k < 10, retraw=True,
                                            **render_kwargs)
            # print(rgb.shape,target_s.shape)
            optimizer.zero_grad()
            loss = img2mse(rgb, target_s)
            loss.backward()
            optimizer.step()

            new_lrate = lrate * (0.8 ** ((k + 1) / 100))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            if (k + 1) % 20 == 0 or k == 0:
                print('Step: ', k)
                print('Loss: ', loss)

                with torch.no_grad():
                    pose_dummy = pose.cpu().detach().numpy()
                    # calculate angles and translation of the optimized pose
                    phi = np.arctan2(pose_dummy[1, 0], pose_dummy[0, 0]) * 180 / np.pi
                    theta = np.arctan2(-pose_dummy[2, 0], np.sqrt(pose_dummy[2, 1] ** 2 + pose_dummy[2, 2] ** 2)) * 180 / np.pi
                    psi = np.arctan2(pose_dummy[2, 1], pose_dummy[2, 2]) * 180 / np.pi
                    translation = np.sqrt(pose_dummy[0,3]**2 + pose_dummy[1,3]**2 + pose_dummy[2,3]**2)
                    #translation = pose_dummy[2, 3]
                    # calculate error between optimized and observed pose
                    phi_error = abs(phi_ref - phi) if abs(phi_ref - phi)<300 else abs(abs(phi_ref - phi)-360)
                    theta_error = abs(theta_ref - theta) if abs(theta_ref - theta)<300 else abs(abs(theta_ref - theta)-360)
                    psi_error = abs(psi_ref - psi) if abs(psi_ref - psi)<300 else abs(abs(psi_ref - psi)-360)
                    rot_error = phi_error + theta_error + psi_error
                    translation_error = abs(translation_ref - translation)
                    print('Reference : ',phi_ref,theta_ref,psi_ref, translation_ref)
                    print('Vals : ',phi,theta,psi,translation)
                    print('Rotation error: ', rot_error)
                    print('Translation error: ', translation_error)
                    print('-----------------------------------')

                if OVERLAY is True:
                    
                    with torch.no_grad():
                        print("Overlaying")
                        rgb, _, _, _ = render(H[i], W[i], focal[i], chunk=args.chunk, c2w=pose[:3, :4], **render_kwargs)
                        print("Overlayed")
                        rgb = rgb.cpu().detach().numpy()
                        rgb8 = to8b(rgb)
                        ref = to8b(obs_img)
                        filename = os.path.join(testsavedir, str(k)+'.png')
                        dst = cv2.addWeighted(rgb8, 0.7, ref, 0.3, 0)
                        imageio.imwrite(filename, dst)
                        imgs.append(dst)
        if OVERLAY is True:
            imageio.mimwrite(os.path.join(testsavedir, 'video.gif'), imgs, fps=8) #quality = 8 for mp4 format
        dists = [8.98,9.83]
        object_pose,obj_dist = getObjectPose((bbox[0][i],bbox[1][i],bbox[2][i],bbox[3][i]),pose.detach().cpu().numpy(),camera_angle,dists[i],W_, H_)
        object_poses.append(object_pose)
        object_dists.append(obj_dist)
    img_vids = []
    itr = 0
    for i in range(2) :
        W[i] = 2*((W[i])//4)
        focal[i] = focal[i]/3
    # H_*=2
    for angle in np.arange(0.,2*math.pi,2*math.pi/64) :
        print(itr)
        world_to_camera = rot_psi(-21.6*math.pi/180.)@np.array([[1.,0.,0.,0.],\
                             [0.,1.,0.,0.],\
                             [0.,0.,1.,8.1],\
                             [0.,0.,0.,1.]])@rot_theta(angle)
        print("World->camera inverse : ", np.linalg.inv(world_to_camera))
        scene_img = np.ones((H_,W_,3))
        for i in range(2) :    
            pose_rel, x, y = getObjectRelPose(object_poses[i],world_to_camera,camera_angle,object_dists[i],W_, H_)
            # print(W[i],focal[i])
            rgb, _, _, _ = render(W[i], W[i], focal[i], chunk=args.chunk, c2w=torch.tensor(pose_rel[:3, :4]).cuda(), **render_kwargs_[i])
            rgb = rgb.cpu().detach().numpy()
            rgb = get_triple_sized(rgb)

            if scene_img[0,0,0] == 1. :
                scene_img*=rgb[0:1,0:1,:]
            # print(np.min(rgb),np.max(rgb))
            W[i]*=3
            if x>-W[i]//2 and y>-W[i]//2 and x<H_+W[i]//2 and y<W_+W[i]//2 :
                scene_img[max(0,x-W[i]//2):min(H_,x+W[i]//2),max(0,y-W[i]//2):min(W_,y+W[i]//2)] \
                    = rgb[-min(0,x-W[i]//2):W[i]+H_-max(H_,x+W[i]//2),-min(0,y-W[i]//2):W[i]+W_-max(W_,y+W[i]//2)]
            W[i] = W[i]//3
        filename = os.path.join("video/", str(itr)+'.png')
        imageio.imwrite(filename, scene_img)
        img_vids.append(scene_img)   
        itr+=1
    
    imageio.mimwrite(os.path.join("video/", 'video.gif'), img_vids, fps=8) #quality = 8 for mp4 format

    
DEBUG = False
OVERLAY = True

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    run()
