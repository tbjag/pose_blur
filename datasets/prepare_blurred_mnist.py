import os
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageFilter
import shutil
from sklearn.model_selection import train_test_split

def prepare_blurred_mnist(data_dir='./data/mnist_all', paired_dir='./data/mnist_all/mnist_blur_pairs'):
    os.makedirs(paired_dir, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)

    for i, (image, _) in enumerate(mnist_dataset):
        pil_image = to_pil_image(image)  # Convert to PIL image
        blurred_image = pil_image.filter(ImageFilter.GaussianBlur(radius=2))  # Apply Gaussian blur

        # Concatenate original and blurred images horizontally
        paired_image = Image.new('L', (pil_image.width * 2, pil_image.height))
        paired_image.paste(pil_image, (0, 0))
        paired_image.paste(blurred_image, (pil_image.width, 0))

        # Save the paired image
        paired_image.save(os.path.join(paired_dir, f'pair_{i}.png'))

    print("Paired dataset created with original and blurred MNIST images.")

def split_dataset(paired_dir='./data/mnist_all/mnist_blur_pairs', train_ratio=0.8):
    train_dir = os.path.join(paired_dir, 'train')
    test_dir = os.path.join(paired_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get all images in the paired directory
    all_images = [f for f in os.listdir(paired_dir) if f.endswith('.png')]
    print(f"Total images in paired directory: {len(all_images)}")

    # Split images into train and test sets
    train_images, test_images = train_test_split(all_images, test_size=1 - train_ratio, random_state=42)

    # Move images to train and test directories
    for image in train_images:
        shutil.move(os.path.join(paired_dir, image), os.path.join(train_dir, image))
    for image in test_images:
        shutil.move(os.path.join(paired_dir, image), os.path.join(test_dir, image))

    print("Dataset organized into train and test folders.")

if __name__ == '__main__':
    data_dir = './data/mnist_all'
    paired_dir = './data/mnist_all/mnist_blur_pairs'
    prepare_blurred_mnist(data_dir, paired_dir)
    split_dataset(paired_dir)
