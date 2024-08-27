from datasets import Features, Dataset, Value, Image, concatenate_datasets
import pandas as pd
import base64
import json
from tqdm import tqdm

import os
import io


from PIL import ImageFile
from PIL import Image as PILImage
ImageFile.LOAD_TRUNCATED_IMAGES = True

from transformers import AutoImageProcessor

image_path = 'E:/scraper-image1024/images/'

def load_data_json():
    # load the data from the json file
    with open('E:/scraper-image1024/data/data.json', 'r') as f:
        data = json.load(f)
    return data

def create_dataset(jsondata):

    image_processor = AutoImageProcessor.from_pretrained("models/ai-detector-vit", use_fast=True)

    # we will create a dataset with two columns: image and label
    # labels can be either 0 or 1, 0 for real images, 1 for ai-created images
    id_counter = 0
    batch_size = 5000
    
    features = Features({
        'image': Image(),
        'label': Value('int32'),
        'id': Value('int32')
    })
    
    dataset = Dataset.from_dict({
        'image': [],
        'label': [],
        'id': []
    }, features=features)


    # initiate the batched loop
    batch_data = {
        'image': [],
        'label': [],
        'id': []
    }

    loop = tqdm(jsondata["ai"])
    for item in loop:
        path = os.path.join(image_path, item['sankaku_id'] + '.jpg')
        try:
            image = PILImage.open(path)
            batch_data["image"].append(image)
            batch_data["label"].append(1)
            batch_data["id"].append(id_counter)
            id_counter += 1

        except Exception as e:
            print('Error opening image:', item['sankaku_id'], path)
            print(e)
            continue

        # add the batch to the dataset
        if len(batch_data["image"]) >= batch_size:
            print('Adding batch to the dataset..')
            dataset = concatenate_datasets([dataset, Dataset.from_dict(batch_data, features=features)])
            batch_data = {
                'image': [],
                'label': [],
                'id': []
            }

    # add the last batch
    dataset = concatenate_datasets([dataset, Dataset.from_dict(batch_data, features=features)])

    print('Created dataset with', len(dataset), 'examples')

    batch_data = {
        'image': [],
        'label': [],
        'id': []
    }

    loop = tqdm(jsondata["real"])
    for item in loop:
        path = os.path.join(image_path, item['sankaku_id'] + '.jpg')
        try:
            image = PILImage.open(path)
            batch_data["image"].append(image)
            batch_data["label"].append(0)
            batch_data["id"].append(id_counter)
            id_counter += 1

        except Exception as e:
            print('Error opening image:', item['sankaku_id'], path)
            print(e)
            continue

        # add the batch to the dataset
        if len(batch_data["image"]) >= batch_size:
            print('Adding batch to the dataset..')
            dataset = concatenate_datasets([dataset, Dataset.from_dict(batch_data, features=features)])
            batch_data = {
                'image': [],
                'label': [],
                'id': []
            }

    # add the last batch
    dataset = concatenate_datasets([dataset, Dataset.from_dict(batch_data, features=features)])
    print('Created dataset with', len(dataset), 'examples')

    return dataset


if __name__ == '__main__':
    data = load_data_json()
    dataset = create_dataset(data)

    # shuffle the dataset
    dataset = dataset.shuffle()

    # split the dataset
    dataset = dataset.train_test_split(test_size=0.005)

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # save the datasets
    num_shards = 20
    for i in range(num_shards):
        shard = train_dataset.shard(num_shards=num_shards, index=i)
        shard.to_parquet(f'data/ai-detector-dataset-hd/train-{i:05d}-of-{num_shards:05d}.parquet')

    test_dataset.to_parquet('data/ai-detector-dataset-hd/test-00000-of-00001.parquet')
    print('Dataset saved successfully!')