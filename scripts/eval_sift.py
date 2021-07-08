# Example usage: python scripts/eval_sift.py --input_dir /data/datasets/kp2d/HPatches/

import argparse
import os
import pickle
import random
import subprocess

import cv2
import numpy as np
import torch
from PIL import Image
from termcolor import colored
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from kp2d.datasets.patches_dataset import PatchesDataset
from kp2d.datasets.wounds_dataset import WoundsDataset
from kp2d.evaluation.evaluate import evaluate_sift
from kp2d.networks.keypoint_net import KeypointNet
from kp2d.networks.keypoint_resnet import KeypointResnet


def main():
    parser = argparse.ArgumentParser(
        description='Script for KeyPointNet testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--input_dir", required=True, type=str, help="Folder containing input images")

    args = parser.parse_args()

    eval_params = [{'res': (320, 240), 'top_k': 300, }]
    eval_params += [{'res': (640, 480), 'top_k': 1000, }]
    eval_params += [{'res': (560,400), 'top_k': 600, }]

    for params in eval_params:
        wound_dataset = WoundsDataset(root_dir=args.input_dir, use_color=True,
                                    output_shape=params['res'])
        #hp_dataset = PatchesDataset(root_dir=args.input_dir, use_color=True,
        #                            output_shape=params['res'], type='a')
        data_loader = DataLoader(wound_dataset,
                                 batch_size=1,
                                 pin_memory=False,
                                 shuffle=False,
                                 num_workers=8,
                                 worker_init_fn=None,
                                 sampler=None)

        print(colored('Evaluating for {} -- top_k {}'.format(params['res'], params['top_k']),'green'))
        rep, loc, c1, c3, c5, mscore, num_points, num_points_50, num_points_70, num_points_90 = evaluate_sift(
            data_loader,
            output_shape=params['res'],
            top_k=params['top_k'],
            use_color=True)

        print('Repeatability {0:.3f}'.format(rep))
        print('Localization Error {0:.3f}'.format(loc))
        print('Correctness d1 {:.3f}'.format(c1))
        print('Correctness d3 {:.3f}'.format(c3))
        print('Correctness d5 {:.3f}'.format(c5))
        print('MScore {:.3f}'.format(mscore))
        print(f'Num Points: {num_points}')
        print(f'Num Points 50% conf: {num_points_50}')
        print(f'Num Points 70% conf: {num_points_70}')
        print(f'Num Points 90% conf: {num_points_90}')


if __name__ == '__main__':
    main()
