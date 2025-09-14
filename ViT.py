#https://github.com/yash3056/Facial_emotion_detection/blob/main/ferplus-vit_V_1.ipynb

import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from transformers import ViTForImageClassification, TrainingArguments, Trainer
import os
import sys

dataset = sys.argv[1]
#dataset = 'fer2013'
#dataset = 'rafdb'

num_labels = 7
emotions = ['happy', 'surprise', 'sad', 'angry', 'disgust', 'fear', 'neutral']

transform_vit = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CustomImageDataset(Dataset):
    def __init__(self, image_folder_dataset, emotions=emotions):
        self.base_ds = image_folder_dataset
        self.desired_classes = emotions
        self.label2id = {cls_name: i for i, cls_name in enumerate(self.desired_classes)}
        self.id2label = {i: cls_name for i, cls_name in enumerate(self.desired_classes)}
    def __len__(self):
        return len(self.base_ds)
    def __getitem__(self, idx):
        img, label = self.base_ds[idx]
        class_name = self.base_ds.classes[label]
        new_label = self.label2id[class_name]
        return {"pixel_values": img, "labels": new_label}

train_dataset = datasets.ImageFolder(root=os.path.join(os.getcwd(), 'datasets', dataset, 'train'), transform=transform_vit)
train_dataset = CustomImageDataset(train_dataset)

model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=num_labels,
    force_download=True,
)
USE_CPU = False
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model.to(device)

training_args = TrainingArguments(
    output_dir='./log',
    num_train_epochs=8,
    per_device_train_batch_size=32,
    save_strategy="epoch",        
    logging_dir='./log',
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_steps=10,            
    save_safetensors=False,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
torch.save(model.state_dict(), os.path.join(os.getcwd(), 'datasets', dataset+'_model_vit.pth'))
