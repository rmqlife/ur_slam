import json
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from utils.pose_util import *

def replace_path(file_path, src, dst):
    directory, filename = os.path.split(file_path)  
    new_filename = filename.replace(src, dst)
    return os.path.join(directory, new_filename)
    
def replace_rgb_to_depth(file_path):
    return replace_path(file_path, 'rgb', 'depth')

def project_to_3d(points, depth, intrinsics, show=True):
    if show:
        plt.imshow(depth)
    
    points_3d = list()
    
    for x,y in points:
        x = math.floor(x) 
        y = math.floor(y)
        d = depth[y][x]        
        # Plot points (x, y) on the image
        if show:
            if d>0:
                plt.scatter(x, y, color='blue', s=10)  # Adjust the size (s) as needed
            else:
                plt.scatter(x, y, color='red', s=10)
        # z = d / depth_scale
        # x = (u - cx) * z / fx
        # y = (v - cy) * z / fy
        # 3d point in meter
        z = d / 1000
        x = (x - intrinsics['cx']) * z / intrinsics['fx'] 
        y = (y - intrinsics['cy']) * z / intrinsics['fy'] 
        
        if show:
            print(f'x:{x} \t y:{y} \t z:{z}')
        points_3d.append((x,y,z))
        
    if show:
        plt.axis('off')  # Turn off axis labels
        plt.show()
    
    return points_3d
    
def load_intrinsics(json_file):
    with open(json_file, "r") as file:
        intrinsic_params = json.load(file)
    return intrinsic_params


def filter_out_zeros_points(pt3d1, pt3d2, threshold=1e-4):
    new_pt3d1 = list()
    new_pt3d2 = list()
    for i in range(len(pt3d1)):
        # print(p)
        p1 = pt3d1[i]
        p2 = pt3d2[i]
        if p1[2]>threshold and p2[2]>threshold:
            new_pt3d1.append(p1)
            new_pt3d2.append(p2)
    return new_pt3d1, new_pt3d2


def print_array(array):
    array = np.round(array, 2)
    print(np.array2string(array, separator=','))


class MyGlue:
    def __init__(self, match_type):
        self.match_type=match_type
        if self.match_type == "LightGlue":
            # import sys
            # sys.path.insert(0,'/home/rmqlife/work/LightGlue')
            from lightglue import LightGlue, SuperPoint, DISK
            from lightglue import viz2d
            import torch
            torch.set_grad_enabled(False)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
            self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)  # load the extractor
            self.matcher = LightGlue(features="superpoint", filter_threshold=0.9).eval().to(self.device)


    def match(self, image0, image1, verbose=True):
        if self.match_type=="LightGlue":
            return self.match_with_lightglue(image0, image1, verbose)
        if self.match_type=="Aruco":
            return self.match_with_aruco(image0, image1, verbose)
        return None, None
    
    def match_with_lightglue(self, image0, image1, verbose):
        from lightglue.utils import numpy_image_to_torch, rbd
        image0 = numpy_image_to_torch(image0)
        image1 = numpy_image_to_torch(image1)

        feats0 = self.extractor.extract(image0.to(self.device))
        feats1 = self.extractor.extract(image1.to(self.device))
        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        m_kpts0 = m_kpts0.cpu().numpy().astype('int')
        m_kpts1 = m_kpts1.cpu().numpy().astype('int')

        return m_kpts0, m_kpts1

    def match_with_aruco(self, image0, image1, verbose):
        from utils.aruco_util import detect_aruco
        corners0, ids0 = detect_aruco(image0, verbose)
        corners1, ids1 = detect_aruco(image1, verbose)
        assert len(ids0)==1
        assert len(ids1)==1
        print('match with aruco', (ids0, ids1))
        # print(corners0[0][0].shape)
        pts0 = corners0[0][0]
        pts1 = corners1[0][0]
        return pts0, pts1

    def match_3d(self, pts0, pts1, depth0, depth1, intrinsics, show=False):
        pt3d0 = project_to_3d(pts0, depth0, intrinsics, show=False)
        pt3d1 = project_to_3d(pts1, depth1, intrinsics, show=False)

        pt3d0, pt3d1 = filter_out_zeros_points(pt3d0, pt3d1)
        pt3d0 = np.array(pt3d0)
        pt3d1 = np.array(pt3d1)

        R, t = find_transformation(pt3d0, pt3d1)
        
        pt3d1_star = transform_poses(R, t, pt3d0)
        err = poses_error(pt3d1_star, pt3d1)
        print('match 3d error', err)
        if show:
            ax = visualize_poses(pt3d0, label="0", color='r',ax=None)
            ax = visualize_poses(pt3d1, label="1", color='b',ax=ax)
            # ax = visualize_poses(pt3d1_star, label="2", color='g',ax=ax)
            # plt.show()
        return R, t

def draw_matches_on_image(img, pts0, pts1):
    # Convert image to RGB (assuming it is in BGR format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create a copy of the image to draw lines on
    img_with_lines = img_rgb.copy()

    # Draw lines between matching points
    for pt0, pt1 in zip(pts0, pts1):
        # Convert points to tuple of integers
        pt0 = tuple(map(int, pt0))
        pt1 = tuple(map(int, pt1))

        # Draw line between corresponding points
        cv2.line(img_with_lines, pt0, pt1, (0, 255, 0), 2)  # Green line with thickness 2

    return img_with_lines

if __name__=="__main__":
    # os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    # glue = MyGlue("LightGlue")
    glue = MyGlue("Aruco")
    id1=9
    id2=66

    image_path1 = f"data/0612-facedown/rgb_{id1}.png"
    image_path2 = f"data/0612-facedown/rgb_{id2}.png"
    rgb1 = cv2.imread(image_path1)
    rgb2 = cv2.imread(image_path2)

    pts0, pts1 = glue.match(rgb1, rgb2,verbose=True)


    depth_path1 = replace_rgb_to_depth(image_path1)
    depth_path2 = replace_rgb_to_depth(image_path2)
    depth1 = cv2.imread(depth_path1, cv2.IMREAD_UNCHANGED)
    depth2 = cv2.imread(depth_path2, cv2.IMREAD_UNCHANGED)

    intrinsics = load_intrinsics("slam_data/intrinsics_d435.json")
    R, t = glue.match_3d(pts0, pts1, depth1, depth2, intrinsics, show=True)
    print(R, t)

    img = draw_matches_on_image(rgb1, pts0, pts1)
    cv2.imshow('ts',img)
    cv2.waitKey(0)
