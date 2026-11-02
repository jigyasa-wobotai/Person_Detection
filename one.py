import os
import shutil

# Folder containing your images
image_folder = r"fallen_person/results/Train/images/partially_visible_low_confidance"

# Folder containing the 5 folders with XML labels
labels_parent_folder = r"fallen_person/falling_person/Train/Annotations/partially_visible_low_confidance"

# Folder where matching XML files will be copied
output_folder = r"fallen_person/results/Train/labels/partially_visible_low_confidance"

os.makedirs(output_folder, exist_ok=True)

# Image extensions
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Search through all 5 folders recursively
for image_file in os.listdir(image_folder):

    image_name, image_ext = os.path.splitext(image_file)

    # Make sure it is an image
    if image_ext.lower() not in image_extensions:
        continue

    # XML label we're looking for
    label_name = image_name + ".xml"

    found = False

    # Search all folders and subfolders
    for root, dirs, files in os.walk(labels_parent_folder):

        if label_name in files:

            source_label = os.path.join(root, label_name)
            destination_label = os.path.join(output_folder, label_name)

            # Copy ONLY the XML
            shutil.copy2(source_label, destination_label)

            print(f"Found: {image_file}  ->  Copied: {label_name}")

            found = True
            break

    if not found:
        print(f"NOT FOUND: {image_file} -> {label_name}")

print("\nDone!")