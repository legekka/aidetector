import argparse
import os

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_from_disk

import tqdm

# Transformation to convert PIL image to Tensor
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

def collate_fn(batch):
    # Convert list of dicts to dict of lists
    batch = {key: [d[key] for d in batch] for key in batch[0]}
    # Apply the transform to each image in the batch
    images = torch.stack([transform(image) for image in batch['image']])
    return images, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the mean and standard deviation of a dataset')
    parser.add_argument('-d', '--dataset', help='Huggingface Dataset path', required=True)
    args = parser.parse_args()

    # Load dataset with huggingface load_dataset
    dataset = load_from_disk(os.path.join(args.dataset, "train"))

    # DataLoader with custom collate_fn to process the images
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_fn)

    mean = 0.0
    std = 0.0
    total_images = 0

    loop = tqdm.tqdm(loader, total=len(loader))

    for images, _ in loop:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  # Reshape to (batch_size, channels, H*W)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

        # show the mean and std live in the progress bar desc, only with 3 decimals
        loop.desc = f'Mean: {mean/total_images}, Std: {std/total_images}'

    mean /= total_images
    std /= total_images

    print(f'Mean: {mean.numpy()}, Std: {std.numpy()}')
