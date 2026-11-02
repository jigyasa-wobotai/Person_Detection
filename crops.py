import os
import csv
import random
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, SiglipProcessor


# ==========================================
# CONFIGURATION
# ==========================================

image_folder = "completely_visible/images/high_confidence"
label_folder = "completely_visible/labels/high_confidence"

output_folder = "siglip_person_crops"
os.makedirs(output_folder, exist_ok=True)

csv_path = "siglip_person_prompt_scores.csv"

SAMPLE_SIZE = 50

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# LOAD SIGLIP2
# ==========================================

print(f"Loading SigLIP2 on {device}")

model_name = "google/siglip2-base-patch16-224"

processor = SiglipProcessor.from_pretrained(model_name)

model = AutoModel.from_pretrained(
    model_name
).to(device)

model.eval()


# ==========================================
# PROMPTS
# ==========================================

candidate_labels = [
    "a person",
    "a partially visible person",
    "a fully visible person"
]


# ==========================================
# GET IMAGE AND LABEL FILES
# ==========================================

images = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith(valid_extensions)
])


labels = sorted([
    f for f in os.listdir(label_folder)
    if f.lower().endswith(".txt")
])


print("Total images:", len(images))
print("Total labels:", len(labels))


assert len(images) == len(labels), (
    "Number of images and labels do not match"
)


# ==========================================
# RANDOM SAMPLE 50 IMAGES
# ==========================================

if SAMPLE_SIZE is not None and len(images) > SAMPLE_SIZE:

    selected_indices = random.sample(
        range(len(images)),
        SAMPLE_SIZE
    )

    selected_indices.sort()

    images = [
        images[i]
        for i in selected_indices
    ]

    labels = [
        labels[i]
        for i in selected_indices
    ]


print("Images selected:", len(images))


# ==========================================
# CSV SETUP
# ==========================================

csv_file = open(
    csv_path,
    "w",
    newline=""
)

writer = csv.writer(csv_file)

writer.writerow([
    "Image",
    "Person_ID",
    "Prompt",
    "Score"
])


# ==========================================
# PROCESS IMAGES
# ==========================================

total_persons = 0


for img_file, label_file in zip(images, labels):

    img_path = os.path.join(
        image_folder,
        img_file
    )

    label_path = os.path.join(
        label_folder,
        label_file
    )


    img = Image.open(img_path).convert("RGB")

    W, H = img.size


    with open(label_path, "r") as f:
        annotations = f.readlines()


    person_id = 0


    for line in annotations:

        values = line.strip().split()

        if len(values) != 5:
            continue


        cls, xc, yc, bw, bh = map(
            float,
            values
        )


        # Only person class
        if int(cls) != 0:
            continue


        person_id += 1
        total_persons += 1


        # ==================================
        # YOLO NORMALIZED TO PIXELS
        # ==================================

        x1 = int((xc - bw / 2) * W)
        y1 = int((yc - bh / 2) * H)

        x2 = int((xc + bw / 2) * W)
        y2 = int((yc + bh / 2) * H)


        # Clamp coordinates

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(W, x2)
        y2 = min(H, y2)


        crop = img.crop(
            (x1, y1, x2, y2)
        )


        # Save crop

        crop_name = (
            f"{os.path.splitext(img_file)[0]}"
            f"_person_{person_id}.jpg"
        )

        crop.save(
            os.path.join(
                output_folder,
                crop_name
            )
        )


        # ==================================
        # SIGLIP2 INFERENCE
        # ==================================

        inputs = processor(
            text=candidate_labels,
            images=crop,
            padding="max_length",
            return_tensors="pt"
        ).to(device)


        with torch.no_grad():

            outputs = model(**inputs)


        # SigLIP2 independent confidence
        scores = torch.sigmoid(
            outputs.logits_per_image
        ).squeeze().cpu().numpy()



        print("\n-----------------------------")
        print(
            f"{img_file} | Person {person_id}"
        )


        for prompt, score in zip(
            candidate_labels,
            scores
        ):

            print(
                f"{prompt:<40} : {score:.5f}"
            )


            writer.writerow([
                img_file,
                person_id,
                prompt,
                float(score)
            ])



csv_file.close()


# ==========================================
# SUMMARY
# ==========================================

print("\n==============================")
print("DONE")
print("==============================")

print(
    "Images processed:",
    len(images)
)

print(
    "Persons processed:",
    total_persons
)

print(
    "Crops saved:",
    output_folder
)

print(
    "CSV saved:",
    csv_path
)