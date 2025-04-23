import os
import shutil
from pathlib import Path

def preprocess_nabirds(base_dir="nabirds", output_dir="nabirds_split"):
    images_dir = os.path.join(base_dir, "images")
    split_file = os.path.join(base_dir, "train_test_split.txt")
    images_txt = os.path.join(base_dir, "images.txt")
    class_labels_file = os.path.join(base_dir, "image_class_labels.txt")
    classes_file = os.path.join(base_dir, "classes.txt")

    for split in ['train', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    print("Reading dataset files...")
    
    
    id_to_file = {}
    with open(images_txt, "r") as f:
        for line in f:
            img_id, filename = line.strip().split()
            id_to_file[img_id] = filename
    
    # map IDs to train/test split (1 = train, 0 = test)
    id_to_split = {}
    with open(split_file, "r") as f:
        for line in f:
            img_id, is_train = line.strip().split()
            id_to_split[img_id] = int(is_train)
    
    # map image IDs to class IDs
    id_to_class = {}
    with open(class_labels_file, "r") as f:
        for line in f:
            img_id, class_id = line.strip().split()
            id_to_class[img_id] = class_id
    
    # map class IDs to class names
    class_id_to_name = {}
    with open(classes_file, "r") as f:
        for line in f:
            class_id, class_name = line.strip().split(maxsplit=1)
            class_id_to_name[class_id] = class_name.replace(" ", "_")
    
    # copy images to the corresponding split and class directories
    print("Copying images to train/test splits...")
    copied_count = 0
    total_images = len(id_to_file)
    
    for img_id, filename in id_to_file.items():
        split = "train" if id_to_split[img_id] == 1 else "test"
        
        class_id = id_to_class[img_id]
        class_name = class_id_to_name[class_id]
    
        class_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        src_path = os.path.join(images_dir, filename)
        dest_path = os.path.join(class_dir, os.path.basename(filename))
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            copied_count += 1
            # check
            if copied_count % 1000 == 0:
                print(f"Progress: {copied_count}/{total_images} images copied")
        else:
            print(f"Warning: Source file not found: {src_path}")
    
    print(f"Done! Copied {copied_count} images to {output_dir}")
    

    train_classes = set()
    test_classes = set()
    
    for split in ['train', 'test']:
        split_dir = os.path.join(output_dir, split)
        classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        if split == 'train':
            train_classes = set(classes)
        else:
            test_classes = set(classes)
            
        class_counts = {}
        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            count = len(os.listdir(class_dir))
            class_counts[class_name] = count
        
        total = sum(class_counts.values())
        print(f"{split} set: {total} images across {len(classes)} classes")
  
    missing_in_train = test_classes - train_classes
    missing_in_test = train_classes - test_classes
    
    if missing_in_train:
        print(f"Warning: {len(missing_in_train)} classes in test set but not in train set")
    if missing_in_test:
        print(f"Warning: {len(missing_in_test)} classes in train set but not in test set")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess NABirds dataset")
    parser.add_argument("--input", type=str, default="nabirds", help="Path to NABirds dataset")
    parser.add_argument("--output", type=str, default="nabirds_split", help="Output directory")
    args = parser.parse_args()
    
    preprocess_nabirds(args.input, args.output)
