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

    # load bad indices from the bad_indices.txt
    bad_indices = []
    with open('bad_indices.txt', 'r') as f:
        for line in f:
            bad_indices.append(int(line.strip()))
    
    # we have to create a new dataset without accessing the items with bad images, because it breaks python.
    # we cannot use the filter method, need to handcraft something
    print(f'Found {len(bad_indices)} bad images')
    print('Length of the dataset before filtering:', len(dataset))
    
    
    # find idx of bad images
    exclude_idx = []
    i = 0
    while i < len(dataset):
        if dataset[i]['id'] in bad_indices:
            exclude_idx.append(i)
        i += 1
        if i % 10000 == 0:
            print(f'Processed {i} images')
    print(f'Got the idx')

    dataset = dataset.select(
        i for i in range(len(dataset)) if i not in set(exclude_idx)
    )
    

    

    # i = 0
    # while i < len(dataset):
    #     if dataset[i]['id'] in bad_indices:
    #         # remove the item from the dataset

    #     else:
    #         i += 1
    #     if i % 10000 == 0:
    #         print(f'Processed {i} images')




    print('Length of the dataset after filtering:', len(dataset))


    num_shards = 20
    for i in range(num_shards):
        shard = dataset.shard(num_shards=num_shards, index=i)
        shard.to_parquet(f'data/ai-detector-dataset/train-{i:05d}-of-{num_shards:05d}.parquet')