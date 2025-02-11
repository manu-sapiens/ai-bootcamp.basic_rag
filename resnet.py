# Import necessary libraries
import requests
from io import BytesIO
import json
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


# ----- 0. Detect GPU if any ----

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    print("CUDA available")
else:
    print('CUDA not available to demonstrate profiling on acceleration devices')


# ----- 1. Accessing and Displaying the Test Image -----

# URL for a sample image (this example uses a publicly available image)
image_url = "https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg"

# Download the image
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(image_url, headers=headers)
print("Status Code:", response.status_code)
print("Content preview:", response.content[:20])

# Display the input image
img = Image.open(BytesIO(response.content)).convert("RGB")

plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.title("Input Image")
plt.axis('off')
plt.show()


# ----- 2. Preprocessing the Image for ResNet-18 -----

# Define the preprocessing pipeline (note the normalization values for ImageNet)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Apply preprocessing
img_tensor = preprocess(img)
# Create a mini-batch as expected by the model
input_batch = img_tensor.unsqueeze(0)



# ----- 3. Running the Model and Displaying the Outputs -----

# Load the pretrained ResNet-18 model
model = models.resnet18(pretrained=True).to(device)
model.eval()  # Set to evaluation mode
input_batch = input_batch.to(device)
# Forward pass through the model
with torch.no_grad():
    output = model(input_batch)



# Download the mapping from class indices to human-readable labels (ImageNet)
class_idx_url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
response = requests.get(class_idx_url)
class_idx = json.loads(response.content)
# Convert the mapping into a more accessible format
idx2label = {int(key): value[1] for key, value in class_idx.items()}



# Get the top 5 predictions
_, indices = torch.sort(output, descending=True)
top5 = indices[0][:5].tolist()

print("Top 5 Predictions:")
for idx in top5:
    print(f"{idx}: {idx2label[idx]}")

# Now go to:
# https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
# and use the python profiler to profile the model