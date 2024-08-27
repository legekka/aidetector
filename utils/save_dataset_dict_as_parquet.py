import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from datasets import load_dataset, DatasetDict, load_from_disk

if __name__ == '__main__':
    dataset = load_from_disk('data/ai-detector-dataset-hd2')
    train_dataset = dataset['train']
    test_dataset = dataset['test']


    num_shards = 20
    for i in range(num_shards):
        shard = train_dataset.shard(num_shards=num_shards, index=i)
        shard.to_parquet(f'data/ai-detector-dataset-hd3/train-{i:05d}-of-{num_shards:05d}.parquet')

    num_shards = 1
    for i in range(num_shards):
        shard = test_dataset.shard(num_shards=num_shards, index=i)
        shard.to_parquet(f'data/ai-detector-dataset-hd3/test-{i:05d}-of-{num_shards:05d}.parquet')

    print('Done')
