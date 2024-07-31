import torch

from transformers import ViTConfig, ViTForImageClassification

config = ViTConfig(
    image_size=512,
    hidden_size=768,
    intermediate_size=3072,
    num_hidden_layers=12,
    num_attention_heads=12,
    patch_size=32,
)
model = ViTForImageClassification(
    config,
)
# set up the id2label and label2id
model.config.id2label = {0: 'ai', 1: 'real'}
model.config.label2id = {'ai': 0, 'real': 1}

# print parameter count 
print('Number of parameters:', model.num_parameters())

# save model to disk
model.save_pretrained('./models/ai-detector-vit')