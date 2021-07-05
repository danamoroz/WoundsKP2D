# Copyright 2020 Toyota Research Institute.  All rights reserved.

import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm

from kp2d.evaluation.descriptor_evaluation import (compute_homography,
                                                   compute_matching_score)
from kp2d.evaluation.detector_evaluation import compute_repeatability
from kp2d.utils.image import to_color_normalized, to_gray_normalized


def evaluate_keypoint_net(data_loader, keypoint_net, output_shape=(320, 240), top_k=300, use_color=True):
    """Keypoint net evaluation script. 

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
    localization_err, repeatability = [], []
    correctness1, correctness3, correctness5, MScore = [], [], [], []

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
            
            data = {'image': sample['image'].numpy().squeeze(),
                    'image_shape' : shape_for_data,
                    'warped_image': sample['warped_image'].numpy().squeeze(),
                    'homography': sample['homography'].squeeze().numpy(),
                    'prob': score_1, 
                    'warped_prob': score_2,
                    'desc': desc1,
                    'warped_desc': desc2}
            
            # Compute repeatabilty and localization error
            _, _, rep, loc_err = compute_repeatability(data, keep_k_points=top_k, distance_thresh=3)
            repeatability.append(rep)
            localization_err.append(loc_err)

            # Compute correctness
            c1, c2, c3 = compute_homography(data, keep_k_points=top_k)
            correctness1.append(c1)
            correctness3.append(c2)
            correctness5.append(c3)

            # Compute matching score
            mscore = compute_matching_score(data, keep_k_points=top_k)
            MScore.append(mscore)

    return np.mean(repeatability), np.mean(localization_err), \
           np.mean(correctness1), np.mean(correctness3), np.mean(correctness5), np.mean(MScore)

def evaluate_sift(data_loader, output_shape=(320, 240), top_k=300, use_color=True):
    """SIFT evaluation script. 

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
    
    conf_threshold = 0.0
    localization_err, repeatability = [], []
    correctness1, correctness3, correctness5, MScore = [], [], [], []
    
    sift = cv2.SIFT_create(nfeatures=1500)


    for i, sample in tqdm(enumerate(data_loader), desc="evaluate_sift"):

        image = sample['image']
        warped_image = sample['warped_image']
        image = np.uint8(image.cpu().squeeze() * 255)
        warped_image = np.uint8(warped_image.cpu().squeeze() * 255)

        im1 = np.moveaxis(image, 0, -1)
        im2 = np.moveaxis(warped_image, 0, -1)

        keypoints1, desc1 = sift.detectAndCompute(im1, None)
        x1 = np.expand_dims(np.array([k.pt[0] for k in keypoints1]), axis=1)
        y1 = np.expand_dims(np.array([k.pt[1] for k in keypoints1]), axis=1)
        probs1 = np.expand_dims(np.array([k.response for k in keypoints1]), axis=1)
        score_1 = np.concatenate((x1, y1, probs1),axis=1)
        desc1 = np.array(desc1)

        keypoints2, desc2 = sift.detectAndCompute(im2, None)
        x2 = np.expand_dims(np.array([k.pt[0] for k in keypoints2]), axis=1)
        y2 = np.expand_dims(np.array([k.pt[1] for k in keypoints2]), axis=1)
        probs2 = np.expand_dims(np.array([k.response for k in keypoints2]), axis=1)
        score_2 = np.concatenate((x2, y2, probs2),axis=1)
        desc2 = np.array(desc2)
            
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
        
        data = {'image': sample['image'].numpy().squeeze(),
                'image_shape' : shape_for_data,
                'warped_image': sample['warped_image'].numpy().squeeze(),
                'homography': sample['homography'].squeeze().numpy(),
                'prob': score_1, 
                'warped_prob': score_2,
                'desc': desc1,
                'warped_desc': desc2}

            
        # Compute repeatabilty and localization error
        _, _, rep, loc_err = compute_repeatability(data, keep_k_points=top_k, distance_thresh=3)
        repeatability.append(rep)
        localization_err.append(loc_err)

        # Compute correctness
        c1, c2, c3 = compute_homography(data, keep_k_points=top_k)
        correctness1.append(c1)
        correctness3.append(c2)
        correctness5.append(c3)

        # Compute matching score
        mscore = compute_matching_score(data, keep_k_points=top_k)
        MScore.append(mscore)

    return np.mean(repeatability), np.mean(localization_err), \
           np.mean(correctness1), np.mean(correctness3), np.mean(correctness5), np.mean(MScore)
