import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vine.src.stega_encoder_decoder import ConditionAdaptor, Decoder
import torch.nn.functional as F

def load_stability_mask(images, predictor):
    with torch.no_grad():
        predictor.eval()
        masks = predictor(images)  # Assuming predictor outputs a mask with values in [0, 1]
        masks = (masks < 0.1).float()  # Thresholding to obtain binary mask
        masks = 1 - masks  # Invert mask: stable regions are 1
        masks = F.interpolate(masks, size=(images.size(2), images.size(3)), mode='bilinear', align_corners=False)
    return masks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
secret_size = 100
batch_size = 32
num_epochs = 10
learning_rate = 1e-4

# Initialize models
condition_adaptor = ConditionAdaptor().to(device)
decoder = Decoder(secret_size=secret_size).to(device)
ckpt_path = None
# Load pre-trained stability predictor
# Replace 'path_to_predictor' with the actual path
predictor = torch.load(os.path.join(ckpt_path, 'ConditionAdaptor.pth'))
predictor.eval()

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(list(condition_adaptor.parameters()) + list(decoder.parameters()), lr=learning_rate)

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for images, _ in train_loader:
        images = images.to(device)
        batch_size = images.size(0)
        secrets = torch.randint(0, 2, (batch_size, secret_size), dtype=torch.float32).to(device)

        # Obtain stability masks
        masks = load_stability_mask(images, predictor)

        # Forward pass
        encoded = condition_adaptor(secrets, images, masks)
        decoded = decoder(encoded)

        # Compute loss
        loss = criterion(decoded, secrets)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(condition_adaptor.state_dict(), 'condition_adaptor_stable_masked.pth')