import os
import shutil

base_dir = "CUB_200_2011"
images_dir = os.path.join(base_dir, "images")
split_file = os.path.join(base_dir, "train_test_split.txt")
images_txt = os.path.join(base_dir, "images.txt")

# Read mappings from image ID to filename and train/test split
id_to_file = {}
with open(images_txt, "r") as f:
    for line in f:
        img_id, filename = line.strip().split()
        id_to_file[img_id] = filename

id_to_split = {}
with open(split_file, "r") as f:
    for line in f:
        img_id, is_train = line.strip().split()
        id_to_split[img_id] = int(is_train)  # 1 = train, 0 = test

# Create output folders
for split in ['train', 'val']:
    os.makedirs(f"cub_split/{split}", exist_ok=True)

# Copy images into the new folder structure
for img_id, filename in id_to_file.items():
    split = "train" if id_to_split[img_id] == 1 else "val"
    class_name = filename.split('/')[0]
    src_path = os.path.join(images_dir, filename)
    dest_dir = os.path.join("cub_split", split, class_name)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(src_path, os.path.join(dest_dir, os.path.basename(filename)))

print("Done")
