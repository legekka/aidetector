from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torch
from PIL import Image

model = AutoModelForImageClassification.from_pretrained("legekka/AI-Anime-Image-Detector-ViT")
feature_extractor = AutoFeatureExtractor.from_pretrained("legekka/AI-Anime-Image-Detector-ViT")

model.eval()

image = Image.open("example.jpg")
inputs = feature_extractor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits

label = model.config.id2label[torch.argmax(logits).item()]
confidence = torch.nn.functional.softmax(logits, dim=1)[0][torch.argmax(logits)].item()

print(f"Prediction: {label} ({round(confidence * 100)}%)")