from yolov8 import WNet
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Load the image
image_path = 'original.jpg'
image = Image.open(image_path).convert('RGB')

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Apply the transformation to the image
x = transform(image).unsqueeze(0)  # Add batch dimension
    
model = WNet()
    
model.load_checkpoint("yolov8n.pt")
model.fuse()

new_image, results = model.inference(x, image_path)


plt.figure(figsize=(20, 10))

plt.title('Official YOLO Detections')
plt.axis('off')


if isinstance(new_image, torch.Tensor):
    im2 = new_image.detach().numpy()
im2 = im2[0,0,:,:]

print(im2.shape)
plt.imshow(im2)
plt.title('Custom Image')
plt.axis('off')
plt.savefig("output_image.png")
plt.show()