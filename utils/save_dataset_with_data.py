import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from datasets import load_dataset, DatasetDict

from huggingface_hub import HfApi, HfFolder


if __name__ == '__main__':
    train_files = os.listdir('data/ai-detector-dataset-hd')
    train_files = [f for f in train_files if 'train' in f]
    train_files = [os.path.join('data/ai-detector-dataset-hd', f) for f in train_files]

    train_dataset = load_dataset('parquet', data_files=train_files, split='train')

    test_files = os.listdir('data/ai-detector-dataset-hd')
    test_files = [f for f in test_files if 'test' in f]
    test_files = [os.path.join('data/ai-detector-dataset-hd', f) for f in test_files]

    test_dataset = load_dataset('parquet', data_files=test_files, split='train')

    dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    # api = HfApi()

    # repo_id = "legekka/ai-detector-dataset-hd"

    # # create the repository
    # #api.create_repo(repo_id, repo_type='dataset', private=True)

    # # push the dataset to the hub
    # dataset.push_to_hub(repo_id, num_shards={"train": 50, "test": 1})

    dataset.save_to_disk('data/ai-detector-dataset-hd2', num_shards={"train": 50, "test": 1}, num_proc=12)