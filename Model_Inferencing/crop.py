# import csv
# import os
# import torch
# from PIL import Image, ImageDraw, ImageFont
# from transformers import pipeline

# # Model
# ckpt = "google/siglip2-base-patch16-224"

# classifier = pipeline(
#     task="zero-shot-image-classification",
#     model=ckpt,
#     device=0 if torch.cuda.is_available() else -1
# )

# # Prompts
# prompts = [
#     "person",
#     "pedestrian"
# ]

# prediction_cnt = {p: 0 for p in prompts}

# crop_folder = "person_crops"

# # output_folder = "pipeline_label_images"
# # os.makedirs(output_folder, exist_ok=True)

# csv_file = "siglip224_head.csv"

# # try:
# #     font = ImageFont.truetype(
# #         "DejaVuSans.ttf",
# #         24
# #     )
# # except Exception:
# #     font = ImageFont.load_default()

# # CSV
# with open(
#     csv_file,
#     "w",
#     newline=""
# ) as f:

#     writer = csv.writer(f)

#     writer.writerow(
#         [
#             "image",
#             "prediction",
#             "head_score",
#             "background_score",
#             "confidence"
#         ]
#     )

#     # Loop through crops
#     for img_name in sorted(
#         os.listdir(crop_folder)
#     ):

#         if not img_name.lower().endswith(
#             (".jpg", ".jpeg", ".png", ".webp")
#         ):
#             continue

#         img_path = os.path.join(
#             crop_folder,
#             img_name
#         )

#         image = Image.open(
#             img_path
#         ).convert("RGB")

#         # Pipeline inference
#         results = classifier(
#             image,
#             candidate_labels=prompts
#         )

#         # Convert results to dictionary
#         scores = {
#             res["label"]: res["score"]
#             for res in results
#         }

#         best_prompt = results[0]["label"]
#         prediction_cnt[best_prompt] += 1
#         confidence = results[0]["score"]*100

#         print(
#             img_name,
#             "=>",
#             best_prompt
#         )

#         # Save scores
#         writer.writerow(
#             [
#                 img_name,
#                 best_prompt,
#                 scores["head"],
#                 scores["background"],
#                 confidence
#             ]
#         )

#         # Save image with only label
#         # draw = ImageDraw.Draw(
#         #     image
#         # )

#         # draw.rectangle(
#         #     (0, 0, 180, 40),
#         #     fill="black"
#         # )

#         # draw.text(
#         #     (5, 5),
#         #     best_prompt,
#         #     fill="lime",
#         #     font=font
#         # )

#         # output_path = os.path.join(
#         #     output_folder,
#         #     img_name
#         # )

#         # image.save(
#         #     output_path
#         # )

# print("\nDone")
# # print("Annotated images:", output_folder)
# print("CSV:", csv_file)

# print("\nPrediction Counts")
# print("-" * 30)

# for label, count in sorted(
#     prediction_cnt.items(),
#     key = lambda x: x[1],
#     reverse=True
# ):
#     print(f"{label:15} {count}")


import csv
import os
import torch
import random
from PIL import Image
from transformers import pipeline


# ============================================================
# Model
# ============================================================

ckpt = "google/siglip2-base-patch16-384"

classifier = pipeline(
    task="zero-shot-image-classification",
    model=ckpt,
    device=0 if torch.cuda.is_available() else -1
)


# ============================================================
# PERSON PROMPTS
# ============================================================

positive_prompts = [
    "a person",
    "a human",
    "a human being",
    "a person standing or walking",
    "a visible human"
]

negative_prompts = [
    "an object with no person",
    "an empty scene with no person",
    "background with no person",
    "a non-human object",
    "a scene without a human"
]

prompts = positive_prompts + negative_prompts


# ============================================================
# Folder
# ============================================================

crop_folder = "person_crops"

# Number of person crops to process
NUM_CROPS = 150


# ============================================================
# CSV
# ============================================================

csv_file = "siglip224_person_HE_score.csv"


with open(
    csv_file,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "image",

            # Individual positive scores
            "positive_1",
            "positive_2",
            "positive_3",
            "positive_4",
            "positive_5",

            # Individual negative scores
            "negative_1",
            "negative_2",
            "negative_3",
            "negative_4",
            "negative_5",

            # Final scores
            "positive_mean",
            "negative_mean",
            "HE_score"
        ]
    )


    # ========================================================
    # Get person crops
    # ========================================================

    img_names = [
        img_name
        for img_name in os.listdir(crop_folder)
        if img_name.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
    ]


    # Take only 150 crops
    img_names = img_names[:NUM_CROPS]


    print(
        f"Processing {len(img_names)} person crops..."
    )


    # ========================================================
    # Process crops
    # ========================================================

    for idx, img_name in enumerate(
        img_names,
        start=1
    ):

        img_path = os.path.join(
            crop_folder,
            img_name
        )

        image = Image.open(
            img_path
        ).convert("RGB")


        # ----------------------------------------------------
        # SigLIP inference
        # ----------------------------------------------------

        results = classifier(
            image,
            candidate_labels=prompts
        )


        # Convert results to dictionary
        scores = {
            res["label"]: res["score"]
            for res in results
        }


        # ----------------------------------------------------
        # Positive scores
        # ----------------------------------------------------

        positive_scores = [
            scores[p]
            for p in positive_prompts
        ]


        # ----------------------------------------------------
        # Negative scores
        # ----------------------------------------------------

        negative_scores = [
            scores[p]
            for p in negative_prompts
        ]


        # ----------------------------------------------------
        # Means
        # ----------------------------------------------------

        positive_mean = (
            sum(positive_scores)
            / len(positive_scores)
        )

        negative_mean = (
            sum(negative_scores)
            / len(negative_scores)
        )


        # ----------------------------------------------------
        # HE Score
        # ----------------------------------------------------

        he_score = (
            positive_mean
            - negative_mean
        )


        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"{idx:3d}/"
            f"{len(img_names)} "
            f"{img_name} "
            f"=> "
            f"Positive Mean: "
            f"{positive_mean:.4f}, "
            f"Negative Mean: "
            f"{negative_mean:.4f}, "
            f"HE: "
            f"{he_score:.4f}"
        )


        # ----------------------------------------------------
        # Save to CSV
        # ----------------------------------------------------

        writer.writerow(
            [
                img_name,

                # Positive scores
                *positive_scores,

                # Negative scores
                *negative_scores,

                # Means
                positive_mean,
                negative_mean,

                # HE
                he_score
            ]
        )


print("\nDone")
print("CSV:", csv_file)

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("siglip224_person_HE_score.csv")

plt.figure(figsize=(12, 6))

plt.scatter(
    range(1, len(df) + 1),
    df["HE_score"],
    s=25,
    alpha=0.7
)

plt.xlabel("Person Crop")
plt.ylabel("HE Score")
plt.title("HE Score for 150 Person Crops")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "person_HE_score_scatter.png",
    dpi=300
)

plt.show()
