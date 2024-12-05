import os
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageFilter
import shutil
from sklearn.model_selection import train_test_split

def prepare_blurred_mnist_for_cyclegan(data_dir='./data/mnist_all', 
                                        output_dir='./data/mnist_split', 
                                        max_images=None, 
                                        blur_radius=2):
    """
    Prepare MNIST dataset for CycleGAN with paired original and blurred images.
    
    Args:
        data_dir (str): Directory to download/find MNIST dataset
        output_dir (str): Output directory for paired images
        max_images (int, optional): Maximum number of images to process
        blur_radius (float): Gaussian blur radius
    """
    # Create directories for trainA and trainB
    trainA_dir = os.path.join(output_dir, 'trainA')
    trainB_dir = os.path.join(output_dir, 'trainB')
    os.makedirs(trainA_dir, exist_ok=True)
    os.makedirs(trainB_dir, exist_ok=True)

    # Transform for MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)

    # Limit number of images if specified
    total_images = len(mnist_dataset) if max_images is None else min(max_images, len(mnist_dataset))

    for i in range(total_images):
        image, _ = mnist_dataset[i]
        pil_image = to_pil_image(image)  # Convert to PIL image
        blurred_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))  # Apply Gaussian blur

        # Save original and blurred images separately
        pil_image.save(os.path.join(trainA_dir, f'img_{i}.png'))
        blurred_image.save(os.path.join(trainB_dir, f'img_{i}.png'))

    print(f"Dataset prepared for CycleGAN with {total_images} images.")

def split_dataset_for_cyclegan(data_dir='./data/mnist_split', train_ratio=0.8):
    """
    Split prepared dataset into train and validation sets.
    
    Args:
        data_dir (str): Directory containing paired images
        train_ratio (float): Proportion of data to use for training
    """
    for domain in ['trainA', 'trainB']:
        domain_dir = os.path.join(data_dir, domain)
        all_images = [f for f in os.listdir(domain_dir) if f.endswith('.png')]
        train_images, val_images = train_test_split(all_images, test_size=1 - train_ratio, random_state=42)

        # Create train and val directories
        train_dir = os.path.join(data_dir, 'train', domain)
        val_dir = os.path.join(data_dir, 'val', domain)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Move files to train and val directories
        for image in train_images:
            shutil.move(os.path.join(domain_dir, image), os.path.join(train_dir, image))
        for image in val_images:
            shutil.move(os.path.join(domain_dir, image), os.path.join(val_dir, image))

    print("CycleGAN dataset split into train and val folders.")

if __name__ == '__main__':
    data_dir = './data/mnist_all'
    output_dir = './data/mnist_blur_pairs'
    
    # Example: Process only 1000 images with blur radius of 3
    prepare_blurred_mnist_for_cyclegan(
        data_dir, 
        output_dir, 
        max_images=1000, 
        blur_radius=3
    )
    split_dataset_for_cyclegan(output_dir)