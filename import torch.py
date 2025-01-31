import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.optim as optim
from PIL import Image
import os

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.features = models.vgg19(pretrained=True).features[:21]
        self.features.eval()  # Ensure the model is in evaluation mode
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for layer in self.features:
            x = layer(x)
            features.append(x)
        return features

class ContentLoss(nn.Module):
    def __init__(self, target, idx):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.idx = idx

    def forward(self, x):
        loss = nn.MSELoss()(x, self.target)
        return loss

class StyleLoss(nn.Module):
    def __init__(self, target, idx):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target).detach()
        self.idx = idx

    def forward(self, x):
        G = self.gram_matrix(x)
        loss = nn.MSELoss()(G, self.target)
        return loss

    def gram_matrix(self, x):
        b, d, h, w = x.size()
        x = x.view(b * d, h * w)
        G = torch.mm(x, x.t())
        return G / (b * d * h * w)

def load_image(image_path, transform=None, max_size=400, shape=None):
    image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB mode
    if max_size:
        size = max_size if max(image.size) > max_size else max(image.size)
        if shape:
            size = shape
        image = transforms.Resize(size)(image)
    if transform:
        image = transform(image).unsqueeze(0)
    return image

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(tensor.device)
    tensor = tensor * std[:, None, None] + mean[:, None, None]
    return tensor

def save_image(tensor, path):
    tensor = denormalize(tensor)
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(path)

def show_image(tensor):
    tensor = denormalize(tensor)
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image)
    image.show()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Correct the file paths
content_img_path = os.path.join("C:", "Users", "Gopal", "OneDrive", "Pictures", "Camera Roll", "sunset-2501727_960_720.webp")
style_img_path = os.path.join("C:", "Users", "Gopal", "OneDrive", "Pictures", "Camera Roll", "monkey-d-luffy-smile.avif")

content_img = load_image(content_img_path, transform)
style_img = load_image(style_img_path, transform, shape=content_img.shape[-2:])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = VGG19().to(device)
content_losses = []
style_losses = []

# Move content_img and style_img to the device before passing to vgg
content_img = content_img.to(device) 
style_img = style_img.to(device)

for idx, layer in enumerate(vgg(content_img)):
    content_losses.append(ContentLoss(layer, idx).to(device))

for idx, layer in enumerate(vgg(style_img)):
    style_losses.append(StyleLoss(layer, idx).to(device))

# Set input image as a copy of the content image
input_img = content_img.clone().requires_grad_(True).to(device)

# Define the optimizer
optimizer = optim.Adam([input_img], lr=0.003)

# Training loop
num_steps = 3000
style_weight = 1e6
for step in range(num_steps):
    optimizer.zero_grad()
    input_features = vgg(input_img)
    content_loss = 0
    style_loss = 0

    for cl in content_losses:
        content_loss += cl(input_features[cl.idx])

    for sl in style_losses:
        style_loss += sl(input_features[sl.idx])

    total_loss = content_loss + style_loss * style_weight
    total_loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {step}, Total loss: {total_loss.item()}")

# Save and display the output image
output_path = os.path.join("C:", "Users", "Gopal", "OneDrive", "Pictures", "Camera Roll", "output_image.jpg")
save_image(input_img, output_path)
print(f"Output image saved at {output_path}")
show_image(input_img)