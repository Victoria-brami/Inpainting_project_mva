import os
import argparse
import json
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../../../test')
    parser.add_argument('--mask_dir', default='../../../masks')
    parser.add_argument('--result_dir', default='../../../cropped_test')
    parser.add_argument('--images_dir', default='../../../cropped_test')
    parser.add_argument('--mask_images_dir', default='../../../mask_images_test')
    parser.add_argument('--img_size', type=int, default=160)
    return parser.parse_args()


def main(args):
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # load dataset
    print('Preprocessing dataset... (it may take a few minutes)')
    print("Image Path", len(os.listdir(args.data_dir)))
    for image_path in os.listdir(args.data_dir):
        #print("Image Path", len(os.listdir(args.data_dir)), os.listdir(args.data_dir))
        img = Image.open(os.path.join(args.data_dir, image_path))
        img = transforms.Resize(args.img_size)(img)
        img = transforms.CenterCrop((args.img_size, args.img_size))(img)
        img = img.save(os.path.join(args.result_dir, image_path))


def couple_imgs_and_masks(args):
    nb_images = len(os.listdir(args.images_dir))
    list_images = os.listdir(args.images_dir)
    list_masks = os.listdir(args.mask_dir)

    if not os.path.exists(args.mask_images_dir):
        os.makedirs(args.mask_images_dir)

    while len(list_images) != 0 and len(list_masks) != 0:

        image_name = list_images.pop(0)
        image_folder = image_name.split('.')[0]
        mask_name = mask.pop(0)

        if not os.path.exists(os.path.join(args.mask_images_dir, image_folder)):
            os.mkdir(os.path.join(args.mask_images_dir, image_folder))

        img = Image.open(os.path.join(args.images_dir, image_name))
        img = img.save(os.path.join(args.mask_images_dir, image_folder, image_name))

        mask = Image.open(os.path.join(args.mask_dir, mask_name))
        mask = img.save(os.path.join(args.mask_images_dir, image_folder, mask_name))

    print(list_images)

if __name__ == '__main__':
    args = parser()
    main(args)
    #couple_imgs_and_masks(args)
