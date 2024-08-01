from datasets import Dataset, load_dataset, Features, Value
from datasets import Image as HuggingFaceImage
from PIL import Image
import io
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from transformers import AutoImageProcessor


if __name__ == '__main__':
    train_files = os.listdir('data/ai-detector-dataset-bad')
    train_files = [f for f in train_files if 'train' in f]
    train_files = [os.path.join('data/ai-detector-dataset-bad', f) for f in train_files]

    dataset = load_dataset('parquet', data_files=train_files, split='train')

    image_processor = AutoImageProcessor.from_pretrained("models/ai-detector-vit", use_fast=True)

    bad_indices = []
    # go through the dataset and check the image integrity
    # import tqdm
    # loop = tqdm.tqdm(dataset)

    # for item in loop:
    #     try:
    #         inputs = image_processor(item['image'], return_tensors="pt")
    #     except:
    #         bad_indices.append(item['id'])
    #         loop.set_description(f'Found {len(bad_indices)} bad images')

    i = 0
    while i < len(dataset):
        try:
            item = dataset[i]
            inputs = image_processor(item['image'], return_tensors="pt")
        except:
            bad_indices.append(item['id'])
            print(f'Found bad image {item["id"]}')
        i += 1
        if i % 10000 == 0:
            print(f'Processed {i} images, found {len(bad_indices)} bad images')

    print(f'Found {len(bad_indices)} bad images')

    # save the bad indices
    with open('bad_indices.txt', 'w') as f:
        for idx in bad_indices:
            f.write(f'{idx}\n')


    # we have to create a new dataset without accessing the items with bad images, because it breaks python.

    num_shards = 20
    for i in range(num_shards):
        shard = dataset.shard(num_shards=num_shards, index=i)
        shard.to_parquet(f'data/ai-detector-dataset/train-{i:05d}-of-{num_shards:05d}.parquet')