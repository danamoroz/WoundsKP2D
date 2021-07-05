from kp2d.evaluation.descriptor_evaluation import select_k_best
import cv2
import numpy as np
from kp2d.utils.image import to_gray_normalized


def draw_matches(data, keep_k_points=1000):
    """
    Compute the matching score between two sets of keypoints with associated descriptors.
    
    Parameters
    ----------
    data: dict
        Input dictionary containing:
        image: numpy.ndarray (W,H)
            Original image
        warped_image: numpy.ndarray (W,H)
            Warped image
        image_shape: tuple (H,W)
            Original image shape.
        homography: numpy.ndarray (3,3)
            Ground truth homography.
        prob: numpy.ndarray (N,3)
            Keypoint vector, consisting of (x,y,probability).
        warped_prob: numpy.ndarray (N,3)
            Warped keypoint vector, consisting of (x,y,probability).
        desc: numpy.ndarray (N,256)
            Keypoint descriptors.
        warped_desc: numpy.ndarray (N,256)
            Warped keypoint descriptors.
        save_path: string
            Path to save the matches image
    keep_k_points: int
        Number of keypoints to select, based on probability.

    """
    shape = data['image_shape']
    real_H = data['homography']
    image = data['image']
    warped_image = data['warped_image']

    # Filter out predictions
    keypoints = data['prob']
    warped_keypoints = data['warped_prob']

    desc = data['desc']
    warped_desc = data['warped_desc']
    
    # Keeps all points for the next frame. The matching for caculating M.Score shouldnt use only in view points.
    keypoints,        desc        = select_k_best(keypoints,               desc, keep_k_points)
    warped_keypoints, warped_desc = select_k_best(warped_keypoints, warped_desc, keep_k_points)
    
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(desc, warped_desc)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    keypoints1 = [cv2.KeyPoint(p[0], p[1], 1) for p in keypoints]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 1) for p in warped_keypoints]

    image = np.uint8(image.cpu().squeeze() * 255)
    warped_image = np.uint8(warped_image.cpu().squeeze() * 255)

    image = np.moveaxis(image, 0, -1)
    warped_image = np.moveaxis(warped_image, 0, -1)
    
    img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    img3 = cv2.drawMatches(img1,keypoints1,img2,keypoints2,matches[:20],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    result = cv2.imwrite(data['save_path'], img3)

    return
