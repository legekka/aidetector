import argparse
import os

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoFeatureExtractor

from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True, help='The path to the huggingface model')
    parser.add_argument('-i', '--image_folder', type=str, required=True, help='The path to the image folder')

    args = parser.parse_args()

    model = AutoModelForImageClassification.from_pretrained(args.model)    
    image_processor = AutoFeatureExtractor.from_pretrained(args.model)

    print('Model loaded successfully')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda:0':
        model = model.to(device)

    model.eval()

    print('Model is ready for inference')

    imagefiles = os.listdir(args.image_folder)
    # select only .jpg files or png files
    imagefiles = [f for f in imagefiles if f.endswith('.jpg') or f.endswith('.png')]
    imagefiles = [os.path.join(args.image_folder, f) for f in imagefiles]


    id2label = model.config.id2label

    for imagefile in imagefiles:
        image = Image.open(imagefile)
        inputs = image_processor(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        # print filename and predicted label with it's probability in percentage (rounded without decimals)
        label = id2label[torch.argmax(logits).item()]
        probability = torch.nn.functional.softmax(logits, dim=1)[0][torch.argmax(logits)].item()
        print(f'{imagefile}: {label} ({round(probability * 100)}%)')
        