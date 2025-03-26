import os
import shutil
import random

def reduce_dataset(input_dir, output_dir, num_samples):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    random.shuffle(files)  # Shuffle to pick random samples
    selected_files = files[:num_samples]

    for file in selected_files:
        shutil.copy(os.path.join(input_dir, file), os.path.join(output_dir, file))

# Paths to trainA and trainB
input_trainA = './data/mnist_blur_pairs/train/trainA'
input_trainB = './data/mnist_blur_pairs/train/trainB'
output_trainA = './data/mnist_blur_pairs_small/trainA'
output_trainB = './data/mnist_blur_pairs_small/trainB'

# Reduce to 160 samples in each domain
reduce_dataset(input_trainA, output_trainA, 160)
reduce_dataset(input_trainB, output_trainB, 160)
