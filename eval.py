import warnings
warnings.filterwarnings("ignore")

import argparse
import os

import torch


from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoFeatureExtractor
from datasets import load_dataset

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from PIL import Image

import tqdm

def eval_transforms(examples):
    # global _eval_transforms
    # examples["pixel_values"] = [_eval_transforms(image.convert('RGB')) for image in examples["image"]]
    examples["pixel_values"] = image_processor(examples["image"], return_tensors='pt')["pixel_values"]
    del examples["image"]
    del examples["id"]
    return examples

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True, help='The path to the huggingface model')

args = parser.parse_args()
image_processor = AutoFeatureExtractor.from_pretrained(args.model, use_fast=True)

size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
_eval_transforms = Compose([Resize(size), ToTensor(), normalize])

if __name__ == '__main__':

    model = AutoModelForImageClassification.from_pretrained(args.model)    
    
    eval_dataset = load_dataset('parquet', data_files='data/ai-detector-dataset-hd/test-00000-of-00001.parquet', split='train')
    eval_dataset = eval_dataset.with_transform(eval_transforms)
    print('Dataset loaded successfully with size:', len(eval_dataset))



    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 8

    if device == 'cuda':
        model = model.to(device)
        batch_size = 64

    model.eval()
    id2label = model.config.id2label


    dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=8)

    accuracy = 0

    for batch in tqdm.tqdm(dataloader):
        images = batch['pixel_values']
        labels = batch['label']

        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            logits = outputs.logits

            for i in range(len(images)):
                original_label = id2label[labels[i].item()]
                predicted_label = id2label[torch.argmax(logits[i]).item()]
                if original_label == predicted_label:
                    accuracy += 1
    accuracy1 = accuracy / len(eval_dataset)
    accuracy2 = 1 - accuracy / len(eval_dataset)
    print('Accuracy 0 Human, 1 AI:', accuracy1)
    print('Accuracy 0 AI, 1 Human:', accuracy2)