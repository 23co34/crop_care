import torch
import torch.nn.functional as F
from PIL import Image
import timm
from pathlib import Path
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dir = Path("/home/nyx/Desktop/project_a1/a12/papaya_split/train")
class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
num_classes = len(class_names)

# 1. Load Model A (EfficientNet-B3)
model_a = timm.create_model("efficientnet_b3", pretrained=False, num_classes=num_classes)
model_a.load_state_dict(torch.load("a12_max_acc_effnet_b3.pth", map_location=device, weights_only=True))
model_a = model_a.to(device)
model_a.eval()

# 2. Load Model B (ConvNeXt V2 Nano)
model_b = timm.create_model("convnextv2_nano", pretrained=False, num_classes=num_classes)
model_b.load_state_dict(torch.load("a12_max_acc_convnext.pth", map_location=device, weights_only=True))
model_b = model_b.to(device)
model_b.eval()

# Standardized transform for both 224x224 models
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# TODO: Point to your real-world test images
folder_path = Path("/home/nyx/Desktop/project_a1/a12/papaya_split/val/Curl")

print(f"\n--- ENSEMBLE PREDICTION RESULTS (EffNet-B3 + ConvNeXt V2) ---")

for img_path in folder_path.iterdir():
    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        try:
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                # Get probabilities from both models
                probs_a = F.softmax(model_a(input_tensor), dim=1)
                probs_b = F.softmax(model_b(input_tensor), dim=1)
                
                # The Ensemble Step: Average the probabilities
                ensemble_probs = (probs_a + probs_b) / 2.0
                
                confidence, predicted_idx = torch.max(ensemble_probs, dim=1)

            predicted_class = class_names[predicted_idx.item()]
            conf_score = confidence.item() * 100

            print(f"File: {img_path.name:<20} | Disease: {predicted_class:<15} | Ensemble Confidence: {conf_score:.2f}%")
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")