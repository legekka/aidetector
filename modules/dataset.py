from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from datasets import load_dataset, concatenate_datasets
import tqdm

class EqualizedDataset(Dataset):
    def __init__(self, dataset_path):
        self.load_datasets(dataset_path)

        self.equalized_dataset = self.__equalize_data()

    def load_datasets(self, dataset_path):
        import os
        # get the list of files in the dataset path
        files = os.listdir(dataset_path)
        train_files = [file for file in files if 'train' in file]
        test_files = [file for file in files if 'test' in file]

        train_files = [os.path.join(dataset_path, file) for file in train_files]
        test_files = [os.path.join(dataset_path, file) for file in test_files]

        self.train_dataset = load_dataset('parquet', data_files=train_files, split='train')
        self.val_dataset = load_dataset('parquet', data_files=test_files, split='train')

        self.train_dataset_ai = self.train_dataset.filter(lambda example: example['label'] == 1)
        self.train_dataset_real = self.train_dataset.filter(lambda example: example['label'] == 0)
  
        del self.train_dataset

    def get_train_dataset(self):
        return self.equalized_dataset

    def get_val_dataset(self):
        return self.val_dataset
       
    def mix_train_dataset(self):
        self.equalized_dataset = self.__equalize_data()
        return self.equalized_dataset

    def __equalize_data(self):
        ai_images_count = len(self.train_dataset_ai)

        self.train_dataset_real = self.train_dataset_real.shuffle()

        # reduce the dataset to the same size as the ai dataset
        real_dataset = self.train_dataset_real.select(range(ai_images_count))

        print("Creating equalized dataset")
        train_dataset = concatenate_datasets([self.train_dataset_ai, real_dataset])
        print("Equalized dataset created")

        train_dataset = train_dataset.shuffle()

        return train_dataset

    # # we will get the number of ai images for base
    # def __equalize_data(self):
    #     # randomize the dataset
    #     self.train_dataset = self.train_dataset.shuffle()

    #     indicies = []

    #     ai_images_count = 0
    #     real_images_count = 0

    #     loop = tqdm.tqdm(range(len(self.train_dataset)), desc='Generating Indicies for Equalized Dataset')
    #     for i in loop:
    #         if self.train_dataset[i]['label'] == 1:
    #             indicies.append(i)
    #             ai_images_count += 1
    #         elif real_images_count < ai_images_count and self.train_dataset[i]['label'] == 0:
    #             indicies.append(i)
    #             real_images_count += 1
        
    #     print('AI images:', ai_images_count)
    #     print('Real images:', real_images_count)

    #     equalized_dataset = {
    #         'image': [],
    #         'label': [],
    #         'id': []
    #     }

    #     loop = tqdm.tqdm(indicies, desc='Creating equalized dataset')
    #     for i in loop:
    #         equalized_dataset['image'].append(self.train_dataset[i]['image'])
    #         equalized_dataset['label'].append(self.train_dataset[i]['label'])
    #         equalized_dataset['id'].append(self.train_dataset[i]['id'])

    #     equalized_dataset = HFDataset.from_dict(equalized_dataset)

    #     # shuffle the dataset
    #     equalized_dataset = equalized_dataset.shuffle()

    #     return equalized_dataset
        