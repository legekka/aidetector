import argparse
import os
import torch
import wandb

from torch.utils.data import DataLoader
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, Resize

from accelerate import Accelerator
from transformers import AutoModelForImageClassification, AutoImageProcessor
from transformers import TrainingArguments, Trainer, TrainerCallback, DefaultDataCollator

from modules.config import Config
from modules.dataset import EqualizedDataset


import PIL.Image as Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import io

def transforms(examples):
    global _transforms
    # for i in range(len(examples["image"])):
    #     if isinstance(examples["image"][i], dict):
    #         examples["pixel_values"].append(_transforms(Image.open(io.BytesIO(examples["image"][i]['bytes'])).convert("RGB")))
    #     elif isinstance(examples["image"][i], Image.Image):
    #         examples["pixel_values"].append(_transforms(examples["image"][i].convert("RGB")))

    # examples["image"] is now already a list of PIL images
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

class CustomTrainerCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        global dataset
        if accelerator.is_main_process:
            print("Mixing the dataset")
        new_train_dataset = dataset.mix_train_dataset()
        new_train_dataset = new_train_dataset.with_transform(transforms)
        model.train_dataset = new_train_dataset

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

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
    _transforms = Compose([RandomResizedCrop(size, scale = (0.66, 1)), ToTensor(), normalize])

    # validation transforms just resizes the image to the model's input size, without cropping
    _val_transforms = Compose([Resize(size), ToTensor(), normalize])

    # Loading the dataset
    global dataset

    dataset = EqualizedDataset(config.dataset_path)
    train_dataset = dataset.get_train_dataset()
    val_dataset = dataset.get_val_dataset()

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[CustomTrainerCallback()],
    )

    if args.wandb and accelerator.is_main_process:
        wandb.init(project=config.wandb['project'], name=config.wandb['name'], tags=config.wandb['tags'])
        wandb.config.update(config._jsonData)
        wandb.watch(model)

    model.config.use_cache = False 

    model, train_dataset, val_dataset = accelerator.prepare(model, train_dataset, val_dataset)

    trainer.train()