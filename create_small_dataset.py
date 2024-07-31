from datasets import Dataset, load_dataset, Features, Value
from datasets import Image as HuggingFaceImage
from PIL import Image
import io

from transformers import AutoImageProcessor


# we are getting when training: ValueError: Invalid image type. Expected either PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray, but got <class 'list'>.
# this is because the image is stored as a dict of bytes, we need to convert it to a PIL image
# dataexample['image']['bytes'] is a byte string, we need to convert it to a PIL image
# we also want to keep the other columns in the dataset

if __name__ == '__main__':
    dataset = load_dataset('parquet', data_files='data/ai-detector-dataset/train-00000-of-00020.parquet', split='train')

    # reduce the dataset to only 1000 examples
    dataset = dataset.select(range(1000))


    import tqdm
    features = Features({
        'image': HuggingFaceImage(),
        'label': Value('int32'),
        'id': Value('int32')
    })
    new_dataset = {
        'image': [],
        'label': [],
        'id': []
    }

    for example in tqdm.tqdm(dataset):
        image = Image.open(io.BytesIO(example['image']['bytes']))
        
        new_dataset['image'].append(image)
        new_dataset['label'].append(example['label'])
        new_dataset['id'].append(example['id'])
    
    print("Creating new dataset")
    new_dataset = Dataset.from_dict(new_dataset, features=features)
    print("New dataset created")
    image_processor = AutoImageProcessor.from_pretrained('models/ai-detector-vit')

    print(new_dataset[0]['image'])

    image = image_processor(new_dataset[0]['image'], return_tensors='pt')
    print(image['pixel_values'].shape)
  
    # save the dataset to the disk
    new_dataset.to_parquet('data/ai-detector-dataset-small/train-00000-of-00001.parquet')
