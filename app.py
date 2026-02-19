import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image

model_data = torch.load(
   "crop_classifier_model.pth",
    map_location="cpu",
    weights_only=False
)

model = models.resnet18(pretrained=False)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    len(model_data["class_to_idx"])
)

model.load_state_dict(model_data["model_state_dict"])
model.eval()

idx_to_class = {
    v:k for k,v in model_data["class_to_idx"].items()
}

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict(image):
    image = transform(image).unsqueeze(0)
    output = model(image)
    _, pred = torch.max(output,1)
    return idx_to_class[pred.item()]

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Crop Classifier"
).launch()
