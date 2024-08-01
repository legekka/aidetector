from datasets import Dataset, load_dataset, Features, Value
from datasets import Image as HuggingFaceImage
from PIL import Image
import io
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from transformers import AutoImageProcessor

def fix_image(example):
    try:
        image = example['image']
    except:
        if example['label'] == 0:
            # open the real image into bytestream to avoid storing paths
            imagebytes = open('tmp/real_image_0_no.jpg', 'rb').read()
            # now open the bytestream as an image
            image = Image.open(io.BytesIO(imagebytes))
            print('Fixed real image')
        else:
            imagebytes = open('tmp/ai_image_1_no.jpg', 'rb').read()
            image = Image.open(io.BytesIO(imagebytes))
            print('Fixed ai image')
            
    example['image'] = image
    example['label'] = example['label']
    example['id'] = example['id']
    return example

# we are getting when training: ValueError: Invalid image type. Expected either PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray, but got <class 'list'>.
# this is because the image is stored as a dict of bytes, we need to convert it to a PIL image
# dataexample['image']['bytes'] is a byte string, we need to convert it to a PIL image
# we also want to keep the other columns in the dataset

if __name__ == '__main__':
    train_files = os.listdir('data/ai-detector-dataset-bad')
    train_files = [f for f in train_files if 'test' in f]
    train_files = [os.path.join('data/ai-detector-dataset-bad', f) for f in train_files]

    dataset = load_dataset('parquet', data_files=train_files, split='train')

    features = dataset.features
    features['image'] = HuggingFaceImage()

    for i in range(len(dataset)):
        if dataset[i]['label'] == 0:
            image = dataset[i]['image']
            image.save('tmp/real_image_0_no.jpg')
            break
    for i in range(len(dataset)):
        if dataset[i]['label'] == 1:
            image = dataset[i]['image']
            image.save('tmp/ai_image_1_no.jpg')
            break

    dataset = dataset.map(fix_image, features=features, num_proc=4)
 
    num_shards = 1
    for i in range(num_shards):
        shard = dataset.shard(num_shards=num_shards, index=i)
        shard.to_parquet(f'data/ai-detector-dataset/test-{i:05d}-of-{num_shards:05d}.parquet')