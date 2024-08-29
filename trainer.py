import argparse
import os
import torch
import wandb
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, Resize

from accelerate import Accelerator
from transformers import AutoModelForImageClassification, AutoImageProcessor
from transformers import TrainingArguments, Trainer, DefaultDataCollator

from modules.config import Config


import PIL.Image as Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_class_weights(dataset):
    import numpy as np
    labels = dataset['label']
    class_counts = np.bincount(labels)

    class_weights = 1.0 / class_counts

    # normalize the weights
    class_weights = class_weights / class_weights.sum() * len(class_counts)

    # convert to tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    return class_weights


def load_parquet_dataset(data_dir, split):
    from datasets import load_dataset
    # check if data_dir exists, if not, we will load the dataset from the hub
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist, loading dataset from the hub")
        return load_dataset(data_dir, split=split)
    else:
        print(f"Loading dataset from {data_dir}")
        parquet_files = os.listdir(data_dir)
        parquet_files = [f for f in parquet_files if split in f]
        parquet_files = [os.path.join(data_dir, f) for f in parquet_files]

        dataset = load_dataset('parquet', data_files=parquet_files, split="train")
        return dataset

def transforms(examples):
    global _transforms
    examples["pixel_values"] = [_transforms(image.convert('RGB')) for image in examples["image"]]


    del examples["image"]
    del examples["id"]
    return examples

def val_transforms(examples):
    global _val_transforms
    examples["pixel_values"] = [_val_transforms(image.convert('RGB')) for image in examples["image"]]
    del examples["image"]
    del examples["id"]
    return examples

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        global loss_fn
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# if we are on windows, we need to check it, and set the torch backend to gloo
if os.name == 'nt':
    try:    
        torch.distributed.init_process_group(backend="gloo")
    except:
        pass

accelerator = Accelerator()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a model to detect AI-created images.')
    parser.add_argument('-c','--config', help='Path to the training config file', required=True)
    parser.add_argument('-w','--wandb', help='Use wandb for logging', action='store_true')
    parser.add_argument('-r','--resume', help='Resume training from the given checkpoint path', default=None)

    args = parser.parse_args()

    config = Config(args.config)

    device = accelerator.device if torch.cuda.is_available() else 'cpu'

    # Loading the model
    if args.resume is not None:
        model = AutoModelForImageClassification.from_pretrained(args.resume)
    else:
        model = AutoModelForImageClassification.from_pretrained(config.model_base)
    model.to(device)


    torch.cuda.empty_cache()

    global image_processor
    global _transforms
    image_processor = AutoImageProcessor.from_pretrained(config.model_base, use_fast=True)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    _transforms = Compose([RandomResizedCrop(size, scale = (0.16, 1)), ToTensor(), normalize])

    # validation transforms just resizes the image to the model's input size, without cropping
    _val_transforms = Compose([Resize(size), ToTensor(), normalize])

    train_dataset = load_parquet_dataset(config.dataset_path, 'train')
    val_dataset = load_parquet_dataset(config.dataset_path, 'test')

    if accelerator.is_main_process:
        print('Train dataset loaded with size:', len(train_dataset))
        print('Validation dataset loaded with size:', len(val_dataset))

    global loss_fn
    class_weights = get_class_weights(train_dataset)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    if accelerator.is_main_process:
        print('Class weights calculated:', class_weights)

    train_dataset = train_dataset.with_transform(transforms)
    val_dataset = val_dataset.with_transform(val_transforms)
    
    # Setting up Trainer

    if config.num_epochs is None:
        num_epochs = config.max_steps * config.batch_size / len(train_dataset)
    else:
        num_epochs = config.num_epochs

    if accelerator.is_main_process:
        print('--- Hyperparameters ---')
        for key in config._jsonData.keys():
            print(f"{key}: {config._jsonData[key]}")
        print('-----------------------')

    training_args = TrainingArguments(
        output_dir=config.checkpoint_path,
        remove_unused_columns=False,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        lr_scheduler_type=config.scheduler,
        optim=config.optimizer,
        learning_rate=config.learning_rate,
        logging_steps=5,
        logging_dir=config.checkpoint_path,
        save_strategy="epoch" if config.num_epochs is not None else "steps",
        save_steps=1000 if config.max_steps is not None else None,
        eval_strategy="epoch" if config.num_epochs is not None else "steps",
        eval_steps=1000 if config.max_steps is not None else None,
        seed=4242,
        bf16=True,
        report_to="wandb" if args.wandb else "none",
        ddp_find_unused_parameters=False,
    )

    data_collator = DefaultDataCollator()


    

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[],
    )

    if args.wandb and accelerator.is_main_process:
        wandb.init(project=config.wandb['project'], name=config.wandb['name'], tags=config.wandb['tags'])
        wandb.config.update(config._jsonData)
        wandb.watch(model)

    model.config.use_cache = False 

    model, train_dataset, val_dataset = accelerator.prepare(model, train_dataset, val_dataset)

    trainer.train()