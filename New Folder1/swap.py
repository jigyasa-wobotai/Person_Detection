# # # # # # Code to remove overlapping class1 labels from label.txt and then annotate images with new label files
# # # # # # import cv2
# # # # # # import shutil
# # # # # # from pathlib import Path


# # # # # # # ============================================================
# # # # # # # DATASET PATHS
# # # # # # # ============================================================

# # # # # # dataset = Path(r"dataset")

# # # # # # images_dir = dataset / "images"
# # # # # # labels_dir = dataset / "labels"


# # # # # # # ============================================================
# # # # # # # RESULTS FOLDERS
# # # # # # # ============================================================

# # # # # # results_dir = dataset / "results"

# # # # # # results_images_dir = results_dir / "images"
# # # # # # results_labels_dir = results_dir / "labels"
# # # # # # results_annotated_dir = results_dir / "annotated"

# # # # # # results_images_dir.mkdir(parents=True, exist_ok=True)
# # # # # # results_labels_dir.mkdir(parents=True, exist_ok=True)
# # # # # # results_annotated_dir.mkdir(parents=True, exist_ok=True)


# # # # # # # ============================================================
# # # # # # # IMAGE EXTENSIONS
# # # # # # # ============================================================

# # # # # # image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# # # # # # # ============================================================
# # # # # # # PROCESS EACH IMAGE
# # # # # # # ============================================================

# # # # # # for image_file in images_dir.iterdir():

# # # # # #     if image_file.suffix.lower() not in image_extensions:
# # # # # #         continue

# # # # # #     print(f"Processing: {image_file.name}")

# # # # # #     # 1. Copy original image
# # # # # #     shutil.copy2(image_file, results_images_dir / image_file.name)

# # # # # #     # 2. Check label existence
# # # # # #     label_file = labels_dir / f"{image_file.stem}.txt"

# # # # # #     if not label_file.exists():
# # # # # #         print(f"  WARNING: No label found for {image_file.name}")
# # # # # #         continue

# # # # # #     # 3. Read original YOLO labels
# # # # # #     with open(label_file, "r") as f:
# # # # # #         lines = f.readlines()

# # # # # #     # 4. Filter out class 1
# # # # # #     corrected_lines = []
# # # # # #     for line in lines:
# # # # # #         parts = line.strip().split()
# # # # # #         if len(parts) != 5:
# # # # # #             continue

# # # # # #         class_id = parts[0]

# # # # # #         # Skip class 1 (hand)
# # # # # #         if class_id == "1":
# # # # # #             continue

# # # # # #         corrected_lines.append(" ".join(parts) + "\n")

# # # # # #     # 5. Save cleaned text label
# # # # # #     with open(results_labels_dir / label_file.name, "w") as f:
# # # # # #         f.writelines(corrected_lines)

# # # # # #     # 6. Read image using OpenCV
# # # # # #     img = cv2.imread(str(image_file))
# # # # # #     if img is None:
# # # # # #         continue

# # # # # #     img_height, img_width = img.shape[:2]

# # # # # #     # Dynamic scaling based on image height
# # # # # #     box_thickness = max(2, int(min(img_width, img_height) * 0.003))
# # # # # #     font_scale = max(0.5, min(img_width, img_height) * 0.001)
# # # # # #     font_thickness = max(1, int(font_scale * 2))

# # # # # #     # 7. Draw annotations
# # # # # #     for line in corrected_lines:
# # # # # #         parts = line.strip().split()

# # # # # #         class_id = int(parts[0])
# # # # # #         x_center = float(parts[1]) * img_width
# # # # # #         y_center = float(parts[2]) * img_height
# # # # # #         box_width = float(parts[3]) * img_width
# # # # # #         box_height = float(parts[4]) * img_height

# # # # # #         x1 = int(x_center - box_width / 2)
# # # # # #         y1 = int(y_center - box_height / 2)
# # # # # #         x2 = int(x_center + box_width / 2)
# # # # # #         y2 = int(y_center + box_height / 2)

# # # # # #         x1, y1 = max(0, x1), max(0, y1)
# # # # # #         x2, y2 = min(img_width - 1, x2), min(img_height - 1, y2)

# # # # # #         # Draw Green Bounding Box
# # # # # #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=box_thickness)

# # # # # #         # Prepare Text
# # # # # #         text = str(class_id)
        
# # # # # #         # Calculate exact text dimensions
# # # # # #         (text_w, text_h), baseline = cv2.getTextSize(
# # # # # #             text, 
# # # # # #             cv2.FONT_HERSHEY_SIMPLEX, 
# # # # # #             font_scale, 
# # # # # #             font_thickness
# # # # # #         )

# # # # # #         # Position label cleanly above box
# # # # # #         text_x = x1
# # # # # #         text_y = y1 - 4
# # # # # #         if text_y - text_h < 0:
# # # # # #             text_y = y1 + text_h + 4

# # # # # #         # Draw Clean Red Number (No background overlap)
# # # # # #         cv2.putText(
# # # # # #             img,
# # # # # #             text,
# # # # # #             (text_x, text_y),
# # # # # #             cv2.FONT_HERSHEY_SIMPLEX,
# # # # # #             fontScale=font_scale,
# # # # # #             color=(0, 0, 255),
# # # # # #             thickness=font_thickness,
# # # # # #             lineType=cv2.LINE_AA
# # # # # #         )

# # # # # #     # 8. Save annotated image
# # # # # #     cv2.imwrite(str(results_annotated_dir / image_file.name), img)


# # # # # # print("\n============================================")
# # # # # # print("DONE!")
# # # # # # print("============================================")
# # # # # # print(f"Images:    {results_images_dir}")
# # # # # # print(f"Labels:    {results_labels_dir}")
# # # # # # print(f"Annotated: {results_annotated_dir}")

# # # # # # Code to replace class1 label with class2 label due to incorrect labelling of head class as 1 rather than 2
# # # import cv2
# # # import shutil
# # # from pathlib import Path


# # # # ============================================================
# # # # DATASET PATHS
# # # # ============================================================

# # # dataset = Path(r"dataset1")

# # # images_dir = dataset / "images"
# # # labels_dir = dataset / "labels"


# # # # ============================================================
# # # # RESULTS FOLDERS
# # # # ============================================================

# # # results_dir = dataset / "results1"

# # # results_images_dir = results_dir / "images"
# # # results_labels_dir = results_dir / "labels"
# # # results_annotated_dir = results_dir / "annotated"

# # # results_images_dir.mkdir(parents=True, exist_ok=True)
# # # results_labels_dir.mkdir(parents=True, exist_ok=True)
# # # results_annotated_dir.mkdir(parents=True, exist_ok=True)


# # # # ============================================================
# # # # IMAGE EXTENSIONS
# # # # ============================================================

# # # image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# # # # ============================================================
# # # # PROCESS EACH IMAGE
# # # # ============================================================

# # # for image_file in images_dir.iterdir():

# # #     if image_file.suffix.lower() not in image_extensions:
# # #         continue

# # #     print(f"Processing: {image_file.name}")

# # #     # 1. Copy original image
# # #     shutil.copy2(
# # #         image_file,
# # #         results_images_dir / image_file.name
# # #     )

# # #     # 2. Check label existence
# # #     label_file = labels_dir / f"{image_file.stem}.txt"

# # #     if not label_file.exists():
# # #         print(f"  WARNING: No label found for {image_file.name}")
# # #         continue

# # #     # 3. Read original YOLO labels
# # #     with open(label_file, "r") as f:
# # #         lines = f.readlines()

# # #     # 4. Replace class 1 with class 2
# # #     corrected_lines = []

# # #     for line in lines:

# # #         parts = line.strip().split()

# # #         if len(parts) != 5:
# # #             continue

# # #         class_id = parts[0]

# # #         # Change class 1 -> class 2
# # #         if class_id == "1":
# # #             parts[0] = "2"

# # #         corrected_lines.append(" ".join(parts) + "\n")

# # #     # 5. Save corrected label
# # #     with open(results_labels_dir / label_file.name, "w") as f:
# # #         f.writelines(corrected_lines)

# # #     # 6. Read image using OpenCV
# # #     img = cv2.imread(str(image_file))

# # #     if img is None:
# # #         continue

# # #     img_height, img_width = img.shape[:2]

# # #     # Dynamic scaling based on image size
# # #     box_thickness = max(
# # #         2,
# # #         int(min(img_width, img_height) * 0.003)
# # #     )

# # #     font_scale = max(
# # #         0.5,
# # #         min(img_width, img_height) * 0.001
# # #     )

# # #     font_thickness = max(
# # #         1,
# # #         int(font_scale * 2)
# # #     )

# # #     # 7. Draw annotations
# # #     for line in corrected_lines:

# # #         parts = line.strip().split()

# # #         class_id = int(parts[0])

# # #         x_center = float(parts[1]) * img_width
# # #         y_center = float(parts[2]) * img_height
# # #         box_width = float(parts[3]) * img_width
# # #         box_height = float(parts[4]) * img_height

# # #         x1 = int(x_center - box_width / 2)
# # #         y1 = int(y_center - box_height / 2)

# # #         x2 = int(x_center + box_width / 2)
# # #         y2 = int(y_center + box_height / 2)

# # #         # Keep coordinates inside image
# # #         x1 = max(0, x1)
# # #         y1 = max(0, y1)

# # #         x2 = min(img_width - 1, x2)
# # #         y2 = min(img_height - 1, y2)

# # #         # Draw green bounding box
# # #         cv2.rectangle(
# # #             img,
# # #             (x1, y1),
# # #             (x2, y2),
# # #             (0, 255, 0),
# # #             thickness=box_thickness
# # #         )

# # #         # Class number
# # #         text = str(class_id)

# # #         # Calculate text dimensions
# # #         (text_w, text_h), baseline = cv2.getTextSize(
# # #             text,
# # #             cv2.FONT_HERSHEY_SIMPLEX,
# # #             font_scale,
# # #             font_thickness
# # #         )

# # #         # Position label above box
# # #         text_x = x1
# # #         text_y = y1 - 4

# # #         if text_y - text_h < 0:
# # #             text_y = y1 + text_h + 4

# # #         # Draw red class number
# # #         cv2.putText(
# # #             img,
# # #             text,
# # #             (text_x, text_y),
# # #             cv2.FONT_HERSHEY_SIMPLEX,
# # #             fontScale=font_scale,
# # #             color=(0, 0, 255),
# # #             thickness=font_thickness,
# # #             lineType=cv2.LINE_AA
# # #         )

# # #     # 8. Save annotated image
# # #     cv2.imwrite(
# # #         str(results_annotated_dir / image_file.name),
# # #         img
# # #     )


# # # print("\n============================================")
# # # print("DONE!")
# # # print("============================================")
# # # print(f"Images:    {results_images_dir}")
# # # print(f"Labels:    {results_labels_dir}")
# # # print(f"Annotated: {results_annotated_dir}")

# # Code to annotate over images using given labels
# import cv2
# from pathlib import Path


# # # # # ============================================================
# # # # # DATASET PATHS
# # # # # ============================================================

# dataset = Path(r"checking/personmulti_final_base_hpo/")

# images_dir = dataset / "images"
# labels_dir = dataset / "labels"

# # Output folder for annotated images only
# annotated_dir = hpo_result / "annotated"
# annotated_dir.mkdir(parents=True, exist_ok=True)


# # ============================================================
# # IMAGE EXTENSIONS
# # ============================================================

# image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# # ============================================================
# # PROCESS EACH IMAGE
# # ============================================================

# for image_file in images_dir.iterdir():

#     if image_file.suffix.lower() not in image_extensions:
#         continue

#     print(f"Processing: {image_file.name}")

#     # --------------------------------------------------------
#     # Read image
#     # --------------------------------------------------------

#     img = cv2.imread(str(image_file))

#     if img is None:
#         print(f"  WARNING: Could not read image")
#         continue

#     img_height, img_width = img.shape[:2]

#     # --------------------------------------------------------
#     # Find corresponding YOLO label
#     # --------------------------------------------------------

#     label_file = labels_dir / f"{image_file.stem}.txt"

#     if not label_file.exists():
#         print(f"  WARNING: No label found")
#         continue

#     # --------------------------------------------------------
#     # Drawing settings
#     # --------------------------------------------------------

#     box_thickness = max(
#         2,
#         int(min(img_width, img_height) * 0.003)
#     )

#     font_scale = max(
#         0.5,
#         min(img_width, img_height) * 0.001
#     )

#     font_thickness = max(
#         1,
#         int(font_scale * 2)
#     )

#     # --------------------------------------------------------
#     # Read labels
#     # --------------------------------------------------------

#     with open(label_file, "r") as f:
#         lines = f.readlines()

#     # --------------------------------------------------------
#     # Draw existing annotations
#     # --------------------------------------------------------

#     for line in lines:

#         parts = line.strip().split()

#         # YOLO format:
#         # class_id x_center y_center width height

#         if len(parts) != 5:
#             continue

#         class_id = parts[0]

#         x_center = float(parts[1]) * img_width
#         y_center = float(parts[2]) * img_height
#         box_width = float(parts[3]) * img_width
#         box_height = float(parts[4]) * img_height

#         # Convert YOLO coordinates to pixel coordinates

#         x1 = int(x_center - box_width / 2)
#         y1 = int(y_center - box_height / 2)

#         x2 = int(x_center + box_width / 2)
#         y2 = int(y_center + box_height / 2)

#         # Keep coordinates inside image

#         x1 = max(0, x1)
#         y1 = max(0, y1)
#         x2 = min(img_width - 1, x2)
#         y2 = min(img_height - 1, y2)

#         # ----------------------------------------------------
#         # Draw bounding box
#         # ----------------------------------------------------

#         cv2.rectangle(
#             img,
#             (x1, y1),
#             (x2, y2),
#             (0, 255, 0),
#             thickness=box_thickness
#         )

#         # ----------------------------------------------------
#         # Draw class ID
#         # ----------------------------------------------------

#         text = str(class_id)

#         (text_w, text_h), baseline = cv2.getTextSize(
#             text,
#             cv2.FONT_HERSHEY_SIMPLEX,
#             font_scale,
#             font_thickness
#         )

#         text_x = x1
#         text_y = y1 - 4

#         # If there isn't room above the box,
#         # put the label inside/below it.

#         if text_y - text_h < 0:
#             text_y = y1 + text_h + 4

#         cv2.putText(
#             img,
#             text,
#             (text_x, text_y),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             font_scale,
#             (0, 0, 255),
#             font_thickness,
#             cv2.LINE_AA
#         )

#     # --------------------------------------------------------
#     # Save annotated image
#     # --------------------------------------------------------

#     output_file = annotated_dir / image_file.name

#     cv2.imwrite(
#         str(output_file),
#         img
#     )


# # ============================================================
# # DONE
# # ============================================================

# print("\n============================================")
# print("DONE!")
# print("============================================")
# print(f"Annotated images saved to: {annotated_dir}")

# # # Code to swap label 1 and label 2 in original label text file
# # from pathlib import Path

# # # ============================================================
# # # LABEL FOLDER
# # # ============================================================

# # LABEL_FOLDER = Path(r"falling_person/Validation/Annotations/fully_visible_high_confidance/5/hand_issue")


# # # ============================================================
# # # SWAP CLASS IDs 1 <-> 2
# # # ============================================================

# # label_files = list(
# #     LABEL_FOLDER.glob("*.txt")
# # )

# # print(
# #     f"Found {len(label_files)} label files."
# # )

# # processed = 0

# # for label_file in label_files:

# #     with open(
# #         label_file,
# #         "r"
# #     ) as f:

# #         lines = f.readlines()

# #     new_lines = []

# #     for line in lines:

# #         # Preserve completely empty lines
# #         if not line.strip():

# #             new_lines.append(line)

# #             continue

# #         parts = line.split()

# #         # Only process valid YOLO lines
# #         if len(parts) == 5:

# #             if parts[0] == "1":

# #                 parts[0] = "2"

# #             elif parts[0] == "2":

# #                 parts[0] = "1"

# #             # Rebuild line
# #             new_lines.append(
# #                 " ".join(parts) + "\n"
# #             )

# #         else:

# #             # Leave unexpected lines unchanged
# #             new_lines.append(line)

# #     # OVERWRITE ORIGINAL FILE
# #     with open(
# #         label_file,
# #         "w"
# #     ) as f:

# #         f.writelines(new_lines)

# #     processed += 1

# #     print(
# #         f"Swapped: {label_file.name}"
# #     )


# # print()
# # print("============================================")
# # print("DONE")
# # print("============================================")
# # print(
# #     f"Files modified: {processed}"
# # )

# Code to draw bbox on images through label files
# import cv2
# from pathlib import Path

# # ============================================================
# # PATH CONFIGURATION
# # ============================================================

# dataset = Path(r"checking/ft2_fully_covered/")

# # Updated to point directly to the 'train' subdirectories
# images_dir = dataset / "images" / "val"
# labels_dir = dataset / "labels" / "val"

# # Output directory for annotated images
# # hpo_result = Path(r"checking/hpo_result/test/annotated")  # Added missing variable definition
# annotated_dir = Path(r"checking/ft2_fully_covered/val")
# annotated_dir.mkdir(parents=True, exist_ok=True)


# # ============================================================
# # IMAGE EXTENSIONS
# # ============================================================

# image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# # ============================================================
# # PROCESS EACH IMAGE
# # ============================================================

# for image_file in images_dir.iterdir():

#     if image_file.suffix.lower() not in image_extensions:
#         continue

#     print(f"Processing: {image_file.name}")

#     # --------------------------------------------------------
#     # Read image
#     # --------------------------------------------------------

#     img = cv2.imread(str(image_file))

#     if img is None:
#         print(f"  WARNING: Could not read image")
#         continue

#     img_height, img_width = img.shape[:2]

#     # --------------------------------------------------------
#     # Find corresponding YOLO label
#     # --------------------------------------------------------

#     label_file = labels_dir / f"{image_file.stem}.txt"

#     if not label_file.exists():
#         print(f"  WARNING: No label found")
#         continue

#     # --------------------------------------------------------
#     # Drawing settings
#     # --------------------------------------------------------

#     box_thickness = max(
#         2,
#         int(min(img_width, img_height) * 0.003)
#     )

#     font_scale = max(
#         0.5,
#         min(img_width, img_height) * 0.001
#     )

#     font_thickness = max(
#         1,
#         int(font_scale * 2)
#     )

#     # --------------------------------------------------------
#     # Read labels
#     # --------------------------------------------------------

#     with open(label_file, "r") as f:
#         lines = f.readlines()

#     # --------------------------------------------------------
#     # Draw existing annotations
#     # --------------------------------------------------------

#     for line in lines:

#         parts = line.strip().split()

#         # YOLO format: class_id x_center y_center width height
#         if len(parts) != 5:
#             continue

#         class_id = parts[0]

#         x_center = float(parts[1]) * img_width
#         y_center = float(parts[2]) * img_height
#         box_width = float(parts[3]) * img_width
#         box_height = float(parts[4]) * img_height

#         # Convert YOLO coordinates to pixel coordinates
#         x1 = int(x_center - box_width / 2)
#         y1 = int(y_center - box_height / 2)
#         x2 = int(x_center + box_width / 2)
#         y2 = int(y_center + box_height / 2)

#         # Keep coordinates inside image boundaries
#         x1 = max(0, x1)
#         y1 = max(0, y1)
#         x2 = min(img_width - 1, x2)
#         y2 = min(img_height - 1, y2)

#         # ----------------------------------------------------
#         # Draw bounding box
#         # ----------------------------------------------------

#         cv2.rectangle(
#             img,
#             (x1, y1),
#             (x2, y2),
#             (0, 255, 0),
#             thickness=box_thickness
#         )

#         # ----------------------------------------------------
#         # Draw class ID
#         # ----------------------------------------------------

#         text = str(class_id)

#         (text_w, text_h), baseline = cv2.getTextSize(
#             text,
#             cv2.FONT_HERSHEY_SIMPLEX,
#             font_scale,
#             font_thickness
#         )

#         text_x = x1
#         text_y = y1 - 4

#         # Adjust text position if it goes off top edge
#         if text_y - text_h < 0:
#             text_y = y1 + text_h + 4

#         cv2.putText(
#             img,
#             text,
#             (text_x, text_y),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             font_scale,
#             (0, 0, 255),
#             font_thickness,
#             cv2.LINE_AA
#         )

#     # --------------------------------------------------------
#     # Save annotated image
#     # --------------------------------------------------------

#     output_file = annotated_dir / image_file.name

#     cv2.imwrite(
#         str(output_file),
#         img
#     )


# # ============================================================
# # DONE
# # ============================================================

# print("\n============================================")
# print("DONE!")
# print("============================================")
# print(f"Annotated images saved to: {annotated_dir}")

# Code to swap label1 with label2 and label2 with label1 
from pathlib import Path

# ============================================================
# LABEL FOLDER
# ============================================================

LABEL_FOLDER = Path(r"checking/final_dataset/labels/New_Folder")


# ============================================================
# SWAP CLASS IDs: 1 -> 2 AND 2 -> 1
# ============================================================

label_files = list(LABEL_FOLDER.glob("*.txt"))

print(f"Found {len(label_files)} label files.")

processed = 0

for label_file in label_files:

    with open(label_file, "r") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        # Preserve empty lines
        if not line.strip():
            new_lines.append(line)
            continue

        parts = line.split()

        # Process valid YOLO format lines
        if len(parts) == 5:
            if parts[0] == "1":
                parts[0] = "2"
            elif parts[0] == "2":
                parts[0] = "1"

            # Rebuild line with single space spacing
            new_lines.append(" ".join(parts) + "\n")
        else:
            # Leave non-standard lines unchanged
            new_lines.append(line)

    # Overwrite the original file
    with open(label_file, "w") as f:
        f.writelines(new_lines)

    processed += 1
    print(f"Swapped: {label_file.name}")

print("\n============================================")
print("DONE")
print("============================================")
print(f"Files modified: {processed}")
