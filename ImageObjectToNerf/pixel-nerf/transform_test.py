import numpy as np
from scipy.spatial.transform import Rotation as R


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

mat = np.array([
                [
                    -0.272744745016098,
                    0.910855233669281,
                    -0.30976295471191406,
                    -0.6195259690284729
                ],
                [
                    -0.9620864391326904,
                    -0.2582210898399353,
                    0.08781562745571136,
                    0.17563126981258392
                ],
                [
                    0.0,
                    0.32197004556655884,
                    0.9467498064041138,
                    1.893499732017517
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ])

print(R.from_matrix(getMatFromPose(
    [90, 0, 179],
    [0, 0, 0]
)[:3, :3]).as_euler('xyz', degrees=True))

print(R.from_matrix(getMatFromPose(
    [90, 0, 179],
    [0, 0, 0]
)[:3, :3]).as_euler('xyz', degrees=True))