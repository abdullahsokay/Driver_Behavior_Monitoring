import os
import random
from PIL import Image
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

dataset_path = "dataset/train/imgs/train"
sample_n = 8

print("=" * 60)
print("IMAGE PROPERTY COMPARISON ACROSS CLASSES")
print("=" * 60)

for folder in sorted(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, folder)
    if not os.path.isdir(folder_path):
        continue
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    sample = random.sample(files, min(sample_n, len(files)))

    sizes = []
    brightness = []
    modes = set()
    for fname in sample:
        img = Image.open(os.path.join(folder_path, fname))
        sizes.append(img.size)  # (width, height)
        modes.add(img.mode)
        arr = np.array(img.convert("RGB"))
        brightness.append(arr.mean())

    unique_sizes = set(sizes)
    print(f"\n{folder}:")
    print(f"  Resolutions seen (sample of {len(sample)}): {unique_sizes}")
    print(f"  Color modes: {modes}")
    print(f"  Avg brightness (0-255): {np.mean(brightness):.1f}")