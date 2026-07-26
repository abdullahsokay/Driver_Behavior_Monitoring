import os
import shutil
from PIL import Image, ImageFilter

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

dataset_path = "dataset/train/imgs/train"
target_folder = os.path.join(dataset_path, "c10")
# IMPORTANT: kept outside dataset_path so Keras' flow_from_directory never
# picks it up as an extra class folder
backup_folder = os.path.join("backups", "c10_original_backup")

# Match the c0-c9 dashcam-dataset characteristics we measured: 320x240, softer/blurrier, JPEG-compressed
target_size = (320, 240)
blur_radius = 0.6
jpeg_quality = 75

if not os.path.exists(target_folder):
    raise FileNotFoundError(f"{target_folder} not found.")

# Back up originals before overwriting anything, so this is reversible
if not os.path.exists(backup_folder):
    print(f"Backing up originals to {backup_folder} ...")
    shutil.copytree(target_folder, backup_folder)
else:
    print(f"Backup already exists at {backup_folder}, skipping backup step.")

files = [f for f in os.listdir(target_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"Processing {len(files)} images in {target_folder} ...")

for i, fname in enumerate(files):
    fpath = os.path.join(target_folder, fname)
    img = Image.open(fpath).convert("RGB")

    # Downscale to the same native resolution as the rest of the dataset, then
    # upscale back so it still fits your existing pipeline's expected size handling
    original_size = img.size
    img = img.resize(target_size, Image.BILINEAR)
    img = img.resize(original_size, Image.BILINEAR)

    # Soften slightly to match the blur/compression profile of the other classes
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Re-save as JPEG at a lower quality to introduce similar compression artifacts
    out_path = os.path.splitext(fpath)[0] + ".jpg"
    img.save(out_path, "JPEG", quality=jpeg_quality)
    if out_path != fpath:
        os.remove(fpath)  # remove original if extension changed (e.g. .png -> .jpg)

    if (i + 1) % 200 == 0:
        print(f"  {i + 1}/{len(files)} done")

print("Done. c10 images normalized. Originals backed up in c10_original_backup/.")