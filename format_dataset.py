from datasets import Dataset, load_dataset
import os

if __name__ == '__main__':
    all_files = os.listdir('ai-detector-dataset-base')

    train_files = [file for file in all_files if 'train' in file]

    print(train_files)

    train_files = [os.path.join('ai-detector-dataset-base', file) for file in train_files]

    dataset = load_dataset('parquet', data_files=train_files, split='train')

    # create a train-test split
    # # shuffle the dataset
    dataset = dataset.shuffle()

    test_dataset = {'image': [], 'label': [], 'id': []}

    ai_images = 0
    real_images = 0
    added_indices = []
    for example in dataset:
        if example['label'] == 1 and ai_images < 500:
            ai_images += 1
            added_indices.append(example['id'])
            test_dataset['id'].append(example['id'])
            test_dataset['image'].append(example['image'])
            test_dataset['label'].append(example['label'])

        elif example['label'] == 0 and real_images < 500:
            real_images += 1
            added_indices.append(example['id'])
            test_dataset['id'].append(example['id'])
            test_dataset['image'].append(example['image'])
            test_dataset['label'].append(example['label'])

        if ai_images == 500 and real_images == 500:
            break

    test_dataset = Dataset.from_dict(test_dataset)        
    
    # remove the added examples from the dataset
    dataset = dataset.filter(lambda example: example['id'] not in added_indices)


    
    print('AI images in the test split:', ai_images)
    print('Real images in the test split:', real_images)


    num_shards = 20
    for i in range(num_shards):
        shard = dataset.shard(num_shards=num_shards, index=i)
        shard.to_parquet(f'ai-detector-dataset/train-{i:05d}-of-{num_shards:05d}.parquet')

    num_shards = 1
    for i in range(num_shards):
        shard = test_dataset.shard(num_shards=num_shards, index=i)
        shard.to_parquet(f'ai-detector-dataset/test-{i:05d}-of-{num_shards:05d}.parquet')