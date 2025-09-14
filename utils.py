import warnings
warnings.filterwarnings("ignore", message=".*flash attention.*")
import os
from sklearn.utils import shuffle
import cv2
import cv2.data
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle
from ResEmoteNet import ResEmoteNet
import shap
from transformers import ViTForImageClassification
from transformers import AutoConfig
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_res = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_vit = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def generate_shap_images(img, model, modelp, device, emotions, pges=[95, 90, 85]):
    if modelp=='resemotenet':
        model.eval()
        img_tensor = transform_res(img).unsqueeze(0).to(device).float()
        img_np = img_tensor.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0)  # (H, W, 1)
        def f(x):
            x_tensor = torch.tensor(x.transpose(0, 3, 1, 2), device=device, dtype=torch.float32)
            with torch.no_grad():
                out = model(x_tensor)
                return F.softmax(out, dim=1).cpu().numpy()
        with torch.no_grad():
            probs = F.softmax(model(img_tensor), dim=1).cpu().numpy().flatten()
        vis_base = np.array(img.copy().resize((64, 64)))
    elif modelp=='vit':
        model.eval()
        img_tensor = transform_vit(img).unsqueeze(0).to(device).float()
        img_np = img_tensor.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0)  # (H, W, C)
        def f(x):
            x_tensor = torch.tensor(x.transpose(0, 3, 1, 2), device=device, dtype=torch.float32)
            with torch.no_grad():
                out = model(pixel_values=x_tensor)
                return torch.softmax(out.logits, dim=-1).detach().cpu().numpy()
        with torch.no_grad():
            outputs = model(pixel_values=img_tensor)
            probs = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy().flatten()
        vis_base = np.array(img.copy().resize((224, 224)))
        
    masker = shap.maskers.Image(mask_value=0, shape=img_np.shape)
    explainer = shap.Explainer(f, masker, output_names=emotions)
    shap_values = explainer(img_np[None, ...])
    
    pred_class = np.argmax(probs)
    shap_map = shap_values.values[0, :, :, :, pred_class]  # (H,W,C)
    pixel_importance = np.mean(shap_map, axis=2)  
    blurred_img = cv2.GaussianBlur(vis_base, (9, 9), 0)

    imgs = {}
    for pge in pges:
        threshold = np.percentile(pixel_importance, pge)
        mask = pixel_importance >= threshold
        masked_img = vis_base.copy()
        masked_img[mask] = 0
        masked_blur = vis_base.copy()
        masked_blur[mask] = blurred_img[mask]
        masked_img = Image.fromarray(masked_img)
        masked_blur = Image.fromarray(masked_blur)
        imgs['mask_'+str(100-pge)] = masked_img
        imgs['blur_'+str(100-pge)] = masked_blur

    return shap_values, probs, imgs

def detect_emotion(image, weights, modelp, emotions, return_model=False, num_labels=7):
    if modelp=='resemotenet' or modelp=='vit':
        if modelp=='resemotenet':
            model = ResEmoteNet().to(device)
            checkpoint = torch.load(weights, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            img_tensor = transform_res(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)
            scores = probabilities.cpu().numpy().flatten()
        elif modelp=='vit':
            config = AutoConfig.from_pretrained(
                'google/vit-base-patch16-224-in21k',
                num_labels=num_labels,
            )
            model = ViTForImageClassification(config).to(device)
            state = torch.load(weights, map_location=device, weights_only=True)
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            if any(k.startswith('module.') for k in state):
                state = {k.replace('module.', ''): v for k, v in state.items()}
            model.load_state_dict(state, strict=True)  # si falta algo, aquí revienta (mejor)
            model.eval()
            img_tensor = transform_vit(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(pixel_values=img_tensor)
                logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            scores = probabilities.detach().cpu().numpy().flatten()
    if return_model:
        return scores, model
    else:
        return scores
