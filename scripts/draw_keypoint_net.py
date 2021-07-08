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
from kp2d.evaluation.evaluate import evaluate_keypoint_net
from kp2d.evaluation.draw import draw_keypoint_net_func
from kp2d.networks.keypoint_net import KeypointNet
from kp2d.networks.keypoint_resnet import KeypointResnet


def main():
    parser = argparse.ArgumentParser(
        description='Script for KeyPointNet testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pretrained_model", type=str, help="pretrained model path")
    parser.add_argument("--input_dir", required=True, type=str, help="Folder containing input images")

    args = parser.parse_args()
    checkpoint = torch.load(args.pretrained_model)
    model_args = checkpoint['config']['model']['params']

    # Check model type
    if 'keypoint_net_type' in checkpoint['config']['model']['params']:
        net_type = checkpoint['config']['model']['params']
    else:
        net_type = KeypointNet # default when no type is specified

    # Create and load keypoint net
    if net_type is KeypointNet:
        keypoint_net = KeypointNet(use_color=model_args['use_color'],
                                do_upsample=model_args['do_upsample'],
                                do_cross=model_args['do_cross'])
    else:
        keypoint_net = KeypointResnet()

    keypoint_net.load_state_dict(checkpoint['state_dict'])
    keypoint_net = keypoint_net.cuda()
    keypoint_net.eval()
    print('Loaded KeypointNet from {}'.format(args.pretrained_model))
    print('KeypointNet params {}'.format(model_args))

    eval_params = [{'res': (320, 240), 'top_k': 300, }] if net_type is KeypointNet else [{'res': (320, 256), 'top_k': 300, }] # KeypointResnet needs (320,256)
    eval_params += [{'res': (640, 480), 'top_k': 1000, }]
    eval_params += [{'res': (560,400), 'top_k': 600, }]

    for params in eval_params:
        #hp_dataset = PatchesDataset(root_dir=args.input_dir, use_color=True,
        #                            output_shape=params['res'], type='a')
        wound_dataset = WoundsDataset(root_dir=args.input_dir, use_color=True,
                                    output_shape=params['res'])
        data_loader = DataLoader(wound_dataset,
                                 batch_size=1,
                                 pin_memory=False,
                                 shuffle=False,
                                 num_workers=8,
                                 worker_init_fn=None,
                                 sampler=None)

        draw_keypoint_net_func(
            data_loader,
            keypoint_net,
            output_shape=params['res'],
            top_k=params['top_k'],
            use_color=True,
            path="matches/wounds/net/r" + str(params['res'][0]))



if __name__ == '__main__':
    main()
