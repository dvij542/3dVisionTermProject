import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__),  "src"))
)

import torch
import torch.nn.functional as F
import numpy as np
import imageio
import src.util as util
import warnings
from src.data import get_split_dataset
from src.render import NeRFRenderer
from src.model import make_model
from scipy.interpolate import CubicSpline
from src.data.SRNDataset import SRNDataset
import tqdm


def extra_args(parser):
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split of data to use train | val | test",
    )
    parser.add_argument(
        "--source",
        "-P",
        type=str,
        default="0",
        help="Source view(s) in image, in increasing order. -1 to do random",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=40,
        help="Number of video frames (rotated views)",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=-10.0,
        help="Elevation angle (negative is above)",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Video scale relative to input size"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.0,
        help="Distance of camera from origin, default is average of z_far, z_near of dataset (only for non-DTU)",
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS of video")
    return parser


args, conf = util.args.parse_args(extra_args)
args.resume = False
args.checkpoints_path = 'checkpoints'
args.name = 'srn_car'


device = util.get_cuda(args.gpu_id[0])

print(args.dataset_format, args.datadir, args.split)

# dset = get_split_dataset(
#     args.dataset_format, args.datadir, want_split=args.split, training=False
# )

datadir = 'mydata'

dset = SRNDataset(datadir, stage='test')
print(len(dset))
data = dset[0]
data_path = data["path"]

print("Data instance loaded:", data_path)

images = data["images"]  #(NV, 3, H, W)
poses = data["poses"]  #(NV, 4, 4)
focal = data["focal"]


if isinstance(focal, float):
    # Dataset implementations are not consistent about
    # returning float or scalar tensor in case of fx=fy
    focal = torch.tensor(focal, dtype=torch.float32)
focal = focal[None]

c = data.get("c")
if c is not None:
    c = c.to(device=device).unsqueeze(0)

NV, _, H, W = images.shape

print("Image size", H, W, "num views", NV)


if args.scale != 1.0:
    Ht = int(H * args.scale)
    Wt = int(W * args.scale)
    if abs(Ht / args.scale - H) > 1e-10 or abs(Wt / args.scale - W) > 1e-10:
        warnings.warn(
            "Inexact scaling, please check {} times ({}, {}) is integral".format(
                args.scale, H, W
            )
        )
    H, W = Ht, Wt


net = make_model(conf["model"]).to(device=device)
print(args.checkpoints_path, args.name)

print("MODEL COMPLETED MAKING>.........")
net.load_weights(args, opt_init = False)

renderer = NeRFRenderer.from_conf(
    conf["renderer"], lindisp=dset.lindisp, eval_batch_size=args.ray_batch_size,
).to(device=device)

render_par = renderer.bind_parallel(net, args.gpu_id, simple_output=True).eval()

# Get the distance from camera to origin
z_near = dset.z_near
z_far = dset.z_far

print("Generating rays")

dtu_format = hasattr(dset, "sub_format") and dset.sub_format == "dtu"


print("Using default (360 loop) camera trajectory")
if args.radius == 0.0:
    radius = (z_near + z_far) * 0.5
    print("> Using default camera radius", radius)
else:
    radius = args.radius

# Use 360 pose sequence from NeRF
render_poses = torch.stack(
    [
        util.pose_spherical(angle, args.elevation, radius)
        for angle in np.linspace(-180, 180, args.num_views + 1)[:-1]
    ],
    0,
)  # (NV, 4, 4)

render_rays = util.gen_rays(
    render_poses,
    W,
    H,
    focal * args.scale,
    z_near,
    z_far,
    c=c * args.scale if c is not None else None,
).to(device=device)
# (NV, H, W, 8)

focal = focal.to(device=device)

source = torch.tensor(list(map(int, args.source.split())), dtype=torch.long)
NS = len(source)
random_source = NS == 1 and source[0] == -1

print(NV, NS, source)
assert not (source >= NV).any()

if renderer.n_coarse < 64:
    # Ensure decent sampling resolution
    renderer.n_coarse = 64
    renderer.n_fine = 128

with torch.no_grad():
    print("Encoding source view(s)")
    if random_source:
        src_view = torch.randint(0, NV, (1,))
    else:
        src_view = source

    net.encode(
        images[src_view].unsqueeze(0),
        poses[src_view].unsqueeze(0).to(device=device),
        focal,
        c=c,
    )

    print("Rendering", args.num_views * H * W, "rays")
    all_rgb_fine = []
    for rays in tqdm.tqdm(
        torch.split(render_rays.view(-1, 8), args.ray_batch_size, dim=0)
    ):
        rgb, _depth = render_par(rays[None])
        all_rgb_fine.append(rgb[0])
    _depth = None
    rgb_fine = torch.cat(all_rgb_fine)
    # rgb_fine (V*H*W, 3)

    frames = rgb_fine.view(-1, H, W, 3)

print("Writing video")
vid_name = ""
if args.split == "test":
    vid_name = "t" + vid_name
elif args.split == "val":
    vid_name = "v" + vid_name
vid_name += "_v" + "_".join(map(lambda x: "{:03}".format(x), source))
vid_path = os.path.join(args.visual_path, args.name, "video" + vid_name + ".mp4")
viewimg_path = os.path.join(
    args.visual_path, args.name, "video" + vid_name + "_view.jpg"
)
# imageio.mimwrite(
#     vid_path, (frames.cpu().numpy() * 255).astype(np.uint8), fps=args.fps, quality=8
# )

img_np = (data["images"][src_view].permute(0, 2, 3, 1) * 0.5 + 0.5).numpy()
img_np = (img_np * 255).astype(np.uint8)
img_np = np.hstack((*img_np,))
imageio.imwrite(viewimg_path, img_np)
imageio.mimsave(vid_path + ".gif", frames.cpu().numpy())

print("Wrote to", vid_path, "view:", viewimg_path)
