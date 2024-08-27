from datasets import Features, Dataset, Value, Image, DatasetDict, concatenate_datasets
import pandas as pd
import base64
import time

from modules.db_model import Post, Tag, PostTag

def get_ai_images_from_db(iteration):
    # get all images that has tag_id 2139147 (tag_id for "ai-created")
    # and post.image512 is not null
    
    print('Starting query..')
    start = time.time()
    # limit the number of images to 10000 and offset by iteration * 10000
    rows = (Post
            .select(Post.image512)
            .join(PostTag)
            .join(Tag)
            .where(Tag.id == 2139147)
            .where(Post.image512.is_null(False))
            .limit(100000)
            .offset(iteration * 100000)
            .dicts()).execute()
    print('Query finished in', time.time() - start, 'seconds')
    print('Number of ai images:', len(rows))

    return rows

def get_real_images_from_db(iteration):
    # to get all the real images, we need to filter all the posts that has the tag_id 2139147
    # the best way for filtering this is to get all the post_id that has the tag_id 2139147 in post_tag table, group them,
    # then in an outer query get all the posts that are not in the group
    rows = (PostTag
            .select(PostTag.post_id)
            .where(PostTag.tag_id == 2139147)
            .group_by(PostTag.post_id)
            .dicts()).execute()
    post_ids = []
    for row in rows:
        post_ids.append(row['post_id'])

    print('Starting query..')
    # get all the posts that are not in the post_ids list
    rows = (Post
            .select(Post.image512)
            .where(Post.id.not_in(post_ids))
            .where(Post.image512.is_null(False))
            .limit(100000)
            .offset(iteration * 100000)
            .dicts()).execute()
    print('Query finished!')
    print('Number of real images:', len(rows))  
    return rows

def create_dataset():

    # we will create a dataset with two columns: image and label
    # labels can be either 0 or 1, 0 for real images, 1 for ai-created images
    id_counter = 0
    features = Features({
        'image': Image(),
        'label': Value('int32'),
        'id': Value('int32')
    })
    
    data = {
        'image': [],
        'label': [],
        'id': []
    }

    i = 0
    count = -1
    while count != 0:
        new_images = get_ai_images_from_db(i)
        for image in new_images:
            data['image'].append({'bytes': base64.b64decode(image['image512'])})
            data['label'].append(1)
            data['id'].append(id_counter)
            id_counter += 1
        count = len(new_images)
        i += 1
    
    ai_images_count = len(data['id'])
    print('Total number of ai images:', ai_images_count)

    i = 0
    while len(data['id']) < 1000000 + ai_images_count:
        new_images = get_real_images_from_db(i)
        for image in new_images:
            data['image'].append({'bytes': base64.b64decode(image['image512'])})
            data['label'].append(0)
            data['id'].append(id_counter)
            id_counter += 1
        i += 1

    print('Total number of real images:', len(data['id']) - ai_images_count)

    print('Creating the dataset..')
    dataset = Dataset.from_dict(data, features=features)

    return dataset

if __name__ == '__main__':
    dataset = create_dataset()

    # randomize the dataset
    dataset = dataset.shuffle()

    print('Splitting the dataset into train and test..')
    desired_test_samples_per_label = 50
    class_0 = dataset.filter(lambda example: example['label'] == 0)
    class_1 = dataset.filter(lambda example: example['label'] == 1)

    test_class_0 = class_0.train_test_split(test_size=desired_test_samples_per_label)['test']
    test_class_1 = class_1.train_test_split(test_size=desired_test_samples_per_label)['test']

    test_dataset = concatenate_datasets([test_class_0, test_class_1]) 

    # based on the ids of the test samples, we will remove them from the dataset
    test_indexes = test_dataset['id']

    # remove the test samples from the dataset
    train_dataset = dataset.filter(lambda example: example['id'] not in test_indexes)

 
    # save the dataset to the disk 
    df_train = train_dataset.to_pandas()
    df_test = test_dataset.to_pandas()

    df_train.to_parquet('ai-detector-dataset/train_dataset.parquet')
    df_test.to_parquet('ai-detector-dataset/test_dataset.parquet')

    print('Dataset saved successfully!')