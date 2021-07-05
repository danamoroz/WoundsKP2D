

import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm

from kp2d.evaluation.descriptor_evaluation import (compute_homography,
                                                   compute_matching_score)
from kp2d.evaluation.detector_evaluation import compute_repeatability
from kp2d.utils.image import to_color_normalized, to_gray_normalized
from kp2d.utils.draw_matches import draw_matches


def draw_keypoint_net_func(data_loader, keypoint_net, output_shape=(320, 240), top_k=300, use_color=True,path=None):
    """Keypoint net draw matches script. 

    Parameters
    ----------
    data_loader: torch.utils.data.DataLoader
        Dataset loader. 
    keypoint_net: torch.nn.module
        Keypoint network.
    output_shape: tuple
        Original image shape.
    top_k: int
        Number of keypoints to use to compute metrics, selected based on probability.    
    use_color: bool
        Use color or grayscale images.
    """
    keypoint_net.eval()
    keypoint_net.training = False

    conf_threshold = 0.0

    with torch.no_grad():
        for i, sample in tqdm(enumerate(data_loader), desc="evaluate_keypoint_net"):
            if use_color:
                image = to_color_normalized(sample['image'].cuda())
                warped_image = to_color_normalized(sample['warped_image'].cuda())
            else:
                image = to_gray_normalized(sample['image'].cuda())
                warped_image = to_gray_normalized(sample['warped_image'].cuda())

            score_1, coord_1, desc1 = keypoint_net(image)
            score_2, coord_2, desc2 = keypoint_net(warped_image)
            B, C, Hc, Wc = desc1.shape

            # Scores & Descriptors
            score_1 = torch.cat([coord_1, score_1], dim=1).view(3, -1).t().cpu().numpy()
            score_2 = torch.cat([coord_2, score_2], dim=1).view(3, -1).t().cpu().numpy()
            desc1 = desc1.view(C, Hc, Wc).view(C, -1).t().cpu().numpy()
            desc2 = desc2.view(C, Hc, Wc).view(C, -1).t().cpu().numpy()

            # Filter based on confidence threshold
            desc1 = desc1[score_1[:, 2] > conf_threshold, :]
            desc2 = desc2[score_2[:, 2] > conf_threshold, :]
            score_1 = score_1[score_1[:, 2] > conf_threshold, :]
            score_2 = score_2[score_2[:, 2] > conf_threshold, :]

            # Prepare data for eval
            if output_shape == None:
                shape_for_data = sample['image'].shape
            else:
                shape_for_data = output_shape
            
            data = {'image': sample['image'],
                    'image_shape' : shape_for_data,
                    'warped_image': sample['warped_image'],
                    'homography': sample['homography'].squeeze().numpy(),
                    'prob': score_1, 
                    'warped_prob': score_2,
                    'desc': desc1,
                    'warped_desc': desc2,
                    'save_path': path + '/m' + str(i) + '.png'}
                    
            
            # Draw matches
            draw_matches(data, keep_k_points=top_k)
            

    return 
