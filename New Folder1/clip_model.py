# # import os
# # import shutil
# # import cv2
# # import torch
# # from PIL import Image
# # from transformers import CLIPProcessor, CLIPModel

# # # Setup device and model (using CLIP to avoid tokenizer config bugs)
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # model_id = "openai/clip-vit-base-patch32"
# # processor = CLIPProcessor.from_pretrained(model_id)
# # model = CLIPModel.from_pretrained(model_id).to(device)

# # def audit_yolo_dataset(images_dir, labels_dir, output_error_dir="image_error", darkness_threshold=45):
# #     # Ensure the 'error' folder exists
# #     os.makedirs(output_error_dir, exist_ok=True)
    
# #     candidate_labels = [
# #         "a person with visible human features", 
# #         "a shadow on the ground", 
# #         "a dark featureless silhouette or background blob"
# #     ]
    
# #     for label_file in os.listdir(labels_dir):
# #         if not label_file.endswith(".txt"):
# #             continue
            
# #         base_name = os.path.splitext(label_file)[0]
# #         image_path = None
# #         image_ext = None
        
# #         # Find the matching original image
# #         for ext in [".jpg", ".png", ".jpeg"]:
# #             potential_path = os.path.join(images_dir, base_name + ext)
# #             if os.path.exists(potential_path):
# #                 image_path = potential_path
# #                 image_ext = ext
# #                 break
                
# #         if not image_path:
# #             continue
            
# #         image_cv = cv2.imread(image_path)
# #         image_pil = Image.open(image_path).convert("RGB")
# #         h, w, _ = image_cv.shape
        
# #         with open(os.path.join(labels_dir, label_file), "r") as f:
# #             lines = f.readlines()
            
# #         has_error = False
        
# #         for line in lines:
# #             parts = line.strip().split()
# #             if not parts:
# #                 continue
                
# #             class_id = int(parts[0])
            
# #             # ONLY PROCESS CLASS 0 (PERSON) - ignore 1 and 2
# #             if class_id != 0:
# #                 continue
                
# #             # YOLO format: class x_center y_center width height (normalized)
# #             x_center, y_center, box_w, box_h = map(float, parts[1:])
            
# #             # Convert to absolute pixel coordinates
# #             x_min = int((x_center - box_w / 2) * w)
# #             y_min = int((y_center - box_h / 2) * h)
# #             x_max = int((x_center + box_w / 2) * w)
# #             y_max = int((y_center + box_h / 2) * h)
            
# #             x_min, y_min = max(0, x_min), max(0, y_min)
# #             x_max, y_max = min(w, x_max), min(h, y_max)
            
# #             if x_max <= x_min or y_max <= y_min:
# #                 continue
                
# #             crop_cv = image_cv[y_min:y_max, x_min:x_max]
# #             crop_pil = image_pil.crop((x_min, y_min, x_max, y_max))
            
# #             # Pixel darkness check (HSV V-channel)
# #             hsv_crop = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2HSV)
# #             mean_brightness = hsv_crop[:, :, 2].mean()
# #             is_excessively_dark = mean_brightness < darkness_threshold
            
# #             # Zero-shot classification check with CLIP
# #             inputs = processor(text=candidate_labels, images=crop_pil, padding=True, return_tensors="pt").to(device)
# #             with torch.no_grad():
# #                 outputs = model(**inputs)
            
# #             # CLIP outputs logits per image for text queries
# #             logits_per_image = outputs.logits_per_image
# #             probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
            
# #             predicted_label = candidate_labels[probs.argmax()]
            
# #             # Flag if predicted as a shadow, featureless blob, or is excessively dark
# #             if predicted_label != "a person with visible human features" or is_excessively_dark:
# #                 has_error = True
# #                 break  # Error found in this image, stop checking other boxes for it
        
# #         # If an error was detected, copy the original clean image into the 'error' folder
# #         if has_error:
# #             dest_path = os.path.join(output_error_dir, base_name + image_ext)
# #             shutil.copy(image_path, dest_path)

# # # Run the function:
# # audit_yolo_dataset("/home/kasm-user/Desktop/folder/personmulti_final_base_hpo/images", "/home/kasm-user/Desktop/folder/personmulti_final_base_hpo/labels")

# # # # # import os
# # # # # import cv2
# # # # # import numpy as np

# # # # # def find_and_save_shadow_images(
# # # # #     image_dir, 
# # # # #     label_dir, 
# # # # #     output_dir="./flagged_shadow_images",
# # # # #     output_txt="flagged_shadow_images.txt",
# # # # #     brightness_threshold=65,  # 0 = pure black, 255 = pure white
# # # # #     shadow_area_scale=0.3     # Height ratio of the strip relative to the person height
# # # # # ):
# # # # #     os.makedirs(output_dir, exist_ok=True)
# # # # #     flagged_images = []
# # # # #     valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    
# # # # #     image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)]
# # # # #     print(f"Scanning {len(image_files)} images for ground shadows...")
    
# # # # #     for img_file in image_files:
# # # # #         img_path = os.path.join(image_dir, img_file)
# # # # #         img = cv2.imread(img_path)
# # # # #         if img is None:
# # # # #             continue
# # # # #         h, w, _ = img.shape
        
# # # # #         txt_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")
# # # # #         if not os.path.exists(txt_path):
# # # # #             continue
            
# # # # #         with open(txt_path, 'r') as f:
# # # # #             lines = f.readlines()
            
# # # # #         has_shadow_profile = False
# # # # #         vis_img = img.copy()
        
# # # # #         for line in lines:
# # # # #             parts = line.strip().split()
# # # # #             if not parts or int(parts[0]) != 0:  # Focus ONLY on class 0 (person)
# # # # #                 continue
                
# # # # #             xc, yc, bw, bh = map(float, parts[1:5])
            
# # # # #             # Define the region directly beneath the person's feet
# # # # #             shadow_yc = yc + (bh / 2) + (shadow_area_scale / 2) * bh
# # # # #             shadow_h = bh * shadow_area_scale
            
# # # # #             # Convert normalized YOLO coordinates to absolute pixel coordinates
# # # # #             x1 = int((xc - bw / 2) * w)
# # # # #             x2 = int((xc + bw / 2) * w)
# # # # #             y1 = int((shadow_yc - shadow_h / 2) * h)
# # # # #             y2 = int((shadow_yc + shadow_h / 2) * h)
            
# # # # #             # Clip bounds
# # # # #             x1, x2 = max(0, x1), min(w, x2)
# # # # #             y1, y2 = max(0, y1), min(h, y2)
            
# # # # #             if x2 <= x1 or y2 <= y1:
# # # # #                 continue
                
# # # # #             # Extract the strip beneath the person and check average brightness
# # # # #             strip = img[y1:y2, x1:x2]
# # # # #             if strip.size == 0:
# # # # #                 continue
                
# # # # #             gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
# # # # #             mean_brightness = np.mean(gray_strip)
            
# # # # #             # If the strip underneath is dark
# # # # #             if mean_brightness < brightness_threshold:
# # # # #                 has_shadow_profile = True
                
# # # # #                 # Draw the person box in Green for reference
# # # # #                 p_x1 = int((xc - bw / 2) * w)
# # # # #                 p_y1 = int((yc - bh / 2) * h)
# # # # #                 p_x2 = int((xc + bw / 2) * w)
# # # # #                 p_y2 = int((yc + bh / 2) * h)
# # # # #                 cv2.rectangle(vis_img, (p_x1, p_y1), (p_x2, p_y2), (0, 255, 0), 2)
                
# # # # #                 # Draw the scanned shadow strip in Red so you can see what triggered it
# # # # #                 cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
# # # # #                 cv2.putText(vis_img, f"Shadow Zone: {int(mean_brightness)}", (x1, max(0, y1 - 5)),
# # # # #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
# # # # #         if has_shadow_profile:
# # # # #             flagged_images.append(img_file)
# # # # #             # Save the visualized image into the output folder
# # # # #             save_path = os.path.join(output_dir, img_file)
# # # # #             cv2.imwrite(save_path, vis_img)
            
# # # # #     # Save text list as well
# # # # #     with open(output_txt, 'w') as f:
# # # # #         f.write("\n".join(flagged_images))
        
# # # # #     print(f"Scan complete! Flagged and saved {len(flagged_images)} images to folder '{output_dir}'.")
# # # # #     print(f"Text list saved to '{output_txt}'.")

# # # # # if __name__ == "__main__":
# # # # #     find_and_save_shadow_images(
# # # # #         image_dir="/home/kasm-user/Desktop/folder/personmulti_final_base_hpo/images/train",
# # # # #         label_dir="/home/kasm-user/Desktop/folder/personmulti_final_base_hpo/labels/train",
# # # # #         output_dir="./flagged_shadow_images",
# # # # #         brightness_threshold=45
# # # # #     )

# # # # import os
# # # # import cv2
# # # # import numpy as np

# # # # def find_standalone_shadow_boxes(
# # # #     image_dir="/home/kasm-user/Desktop/folder/personmulti_final_base_hpo/images/train",
# # # #     label_dir="/home/kasm-user/Desktop/folder/personmulti_final_base_hpo/labels/train",
# # # #     output_dir="./standalone_shadow_images",
# # # #     output_txt="standalone_shadow_images.txt",
# # # #     isolation_distance_threshold=0.25 # How far away a box must be from others to be considered "isolated"
# # # # ):
# # # #     os.makedirs(output_dir, exist_ok=True)
# # # #     flagged_images = []
# # # #     valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    
# # # #     if not os.path.exists(image_dir) or not os.path.exists(label_dir):
# # # #         print(f"Error: Directory paths do not exist. Please verify your paths.")
# # # #         print(f"Images: {image_dir}")
# # # #         print(f"Labels: {label_dir}")
# # # #         return

# # # #     image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)]
# # # #     print(f"Scanning {len(image_files)} training images for isolated/shadow boxes...")
    
# # # #     for img_file in image_files:
# # # #         img_path = os.path.join(image_dir, img_file)
# # # #         img = cv2.imread(img_path)
# # # #         if img is None:
# # # #             continue
# # # #         h, w, _ = img.shape
        
# # # #         txt_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")
# # # #         if not os.path.exists(txt_path):
# # # #             continue
            
# # # #         with open(txt_path, 'r') as f:
# # # #             lines = f.readlines()
            
# # # #         person_boxes = []
# # # #         for line in lines:
# # # #             parts = line.strip().split()
# # # #             if not parts or int(parts[0]) != 0:  # Only focus on class 0 (person)
# # # #                 continue
# # # #             xc, yc, bw, bh = map(float, parts[1:5])
# # # #             person_boxes.append((xc, yc, bw, bh))
            
# # # #         # If there are no person boxes, skip
# # # #         if not person_boxes:
# # # #             continue
            
# # # #         has_isolated_box = False
# # # #         vis_img = img.copy()
# # # #         box_statuses = [] # Track which boxes are normal vs isolated
        
# # # #         # Check each box to see if it is far away from all other boxes (isolated)
# # # #         for i, box_a in enumerate(person_boxes):
# # # #             is_isolated = True
# # # #             for j, box_b in enumerate(person_boxes):
# # # #                 if i == j:
# # # #                     continue
# # # #                 # Calculate distance between centers (normalized coordinates)
# # # #                 dist = np.sqrt((box_a[0] - box_b[0])**2 + (box_a[1] - box_b[1])**2)
# # # #                 if dist < isolation_distance_threshold:
# # # #                     is_isolated = False
# # # #                     break
            
# # # #             box_statuses.append((box_a, is_isolated))
# # # #             if is_isolated:
# # # #                 has_isolated_box = True
                
# # # #         # If an isolated box is found, draw and save the image
# # # #         if has_isolated_box:
# # # #             flagged_images.append(img_file)
            
# # # #             for box, is_isolated in box_statuses:
# # # #                 xc, yc, bw, bh = box
# # # #                 x1 = int((xc - bw / 2) * w)
# # # #                 y1 = int((yc - bh / 2) * h)
# # # #                 x2 = int((xc + bw / 2) * w)
# # # #                 y2 = int((yc + bh / 2) * h)
                
# # # #                 if is_isolated:
# # # #                     # Draw isolated/potential shadow box in RED
# # # #                     cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
# # # #                     cv2.putText(vis_img, "Isolated / Shadow?", (x1, max(0, y1 - 5)),
# # # #                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
# # # #                 else:
# # # #                     # Draw normal grouped person box in GREEN
# # # #                     cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
# # # #                     cv2.putText(vis_img, "Person", (x1, max(0, y1 - 5)),
# # # #                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
# # # #             save_path = os.path.join(output_dir, img_file)
# # # #             cv2.imwrite(save_path, vis_img)
            
# # # #     # Save text list of flagged images
# # # #     with open(output_txt, 'w') as f:
# # # #         f.write("\n".join(flagged_images))
        
# # # #     print(f"Scan complete! Found {len(flagged_images)} images with isolated/shadow boxes.")
# # # #     print(f"Saved visualization images to '{output_dir}' and text list to '{output_txt}'.")

# # # # if __name__ == "__main__":
# # # #     find_standalone_shadow_boxes()

# # # import os
# # # import shutil
# # # import torch
# # # from pathlib import Path
# # # from PIL import Image
# # # from tqdm import tqdm
# # # from transformers import CLIPProcessor, CLIPModel

# # # # ==================== CONFIGURATION ====================
# # # IMAGE_DIR = Path("personmulti_final_base_hpo/images")  
# # # LABEL_DIR = Path("personmulti_final_base_hpo/labels")  

# # # # Output directory structure for isolated errors
# # # OUTPUT_DIR = Path("images_error")
# # # OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
# # # OUTPUT_LABEL_DIR = OUTPUT_DIR / "labels"
# # # OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
# # # OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

# # # PERSON_CLASS_ID = 0  # Only focus on class 0 (Person), ignore classes 1, 2, etc.

# # # # Detection Thresholds
# # # DARKNESS_BRIGHTNESS_THRESHOLD = 15.0  # Average pixel brightness below this is pitch black/unviewable
# # # ERROR_CONFIDENCE_THRESHOLD = 0.35     # Threshold for shadow/camouflage similarity score
# # # # =======================================================

# # # def load_person_yolo_boxes(label_path, img_width, img_height):
# # #     """Parses YOLO label files: keeps ONLY class 0 (Person) and skips classes 1, 2, etc."""
# # #     boxes = []
# # #     if not label_path.exists():
# # #         return boxes
    
# # #     with open(label_path, "r") as f:
# # #         for line in f.readlines():
# # #             parts = line.strip().split()
# # #             if not parts:
# # #                 continue
            
# # #             try:
# # #                 cls_id = int(parts[0])
# # #             except ValueError:
# # #                 continue
            
# # #             # Skip any class that is NOT a person
# # #             if cls_id != PERSON_CLASS_ID:
# # #                 continue
            
# # #             # Parse YOLO normalized coordinates (x_center, y_center, w, h)
# # #             if len(parts) >= 5:
# # #                 x_center, y_center, w, h = map(float, parts[1:5])
# # #                 x1 = int((x_center - w / 2) * img_width)
# # #                 y1 = int((y_center - h / 2) * img_height)
# # #                 x2 = int((x_center + w / 2) * img_width)
# # #                 y2 = int((y_center + h / 2) * img_height)
                
# # #                 # Clip coordinates to image boundaries
# # #                 x1, y1 = max(0, x1), max(0, y1)
# # #                 x2, y2 = min(img_width, x2), min(img_height, y2)
                
# # #                 if x2 > x1 and y2 > y1:
# # #                     boxes.append((x1, y1, x2, y2))
# # #     return boxes

# # # def is_image_too_dark(pil_img):
# # #     """Checks if an image has a pitch-black or unviewable dark background."""
# # #     grayscale = pil_img.convert("L")
# # #     stat = torch.tensor(list(grayscale.getdata())).float()
# # #     avg_brightness = stat.mean().item()
# # #     return avg_brightness < DARKNESS_BRIGHTNESS_THRESHOLD

# # # def run_clip_error_filtering():
# # #     print("=" * 50)
# # #     print("Initializing CLIP Model (openai/clip-vit-base-patch32)...")
# # #     device = "cuda" if torch.cuda.is_available() else "cpu"
# # #     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# # #     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# # #     print(f"CLIP Model loaded successfully on {device}.\n")

# # #     valid_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
# # #     image_files = sorted([p for p in IMAGE_DIR.glob("**/*") if p.suffix in valid_extensions])
# # #     print(f"Total dataset images to evaluate: {len(image_files)}")

# # #     error_count = 0
# # #     skipped_count = 0

# # #     # Prompts focused STRICTLY on identifying errors (shadows & camouflage)
# # #     error_prompts = [
# # #         "a human shadow on the ground or wall",
# # #         "a person camouflaged blending completely into background",
# # #         "a normal clear photo of a person"  # Baseline reference to contrast against errors
# # #     ]

# # #     for img_path in tqdm(image_files, desc="Filtering Error Images"):
# # #         # Image and label names match by stem (e.g., image.jpg -> image.txt)
# # #         label_path = LABEL_DIR / f"{img_path.stem}.txt"
        
# # #         try:
# # #             pil_img = Image.open(img_path).convert("RGB")
# # #         except Exception:
# # #             continue
            
# # #         w, h = pil_img.size
# # #         is_error = False

# # #         # 1. Check for pitch-black / unviewable background condition
# # #         if is_image_too_dark(pil_img):
# # #             is_error = True
# # #         else:
# # #             # Load only person bounding boxes (class 0)
# # #             boxes = load_person_yolo_boxes(label_path, w, h)
            
# # #             if not boxes:
# # #                 skipped_count += 1
# # #                 continue

# # #             # 2. Crop bounding boxes and evaluate using error-specific prompts
# # #             for box in boxes:
# # #                 cropped_img = pil_img.crop(box)
                
# # #                 if cropped_img.width < 10 or cropped_img.height < 10:
# # #                     continue

# # #                 inputs = processor(
# # #                     text=error_prompts, 
# # #                     images=cropped_img, 
# # #                     return_tensors="pt", 
# # #                     padding=True
# # #                 ).to(device)

# # #                 with torch.no_grad():
# # #                     outputs = model(**inputs)
# # #                     probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

# # #                 # Index mapping from error_prompts:
# # #                 # 0: "a human shadow..."
# # #                 # 1: "a person camouflaged..."
# # #                 # 2: "a normal clear photo of a person"
                
# # #                 shadow_score = probs[0]
# # #                 camouflage_score = probs[1]
# # #                 clear_person_score = probs[2]

# # #                 # Flag as error if shadow or camouflage score is high and beats the clear person score
# # #                 if (shadow_score > ERROR_CONFIDENCE_THRESHOLD and shadow_score > clear_person_score) or \
# # #                    (camouflage_score > ERROR_CONFIDENCE_THRESHOLD and camouflage_score > clear_person_score):
# # #                     is_error = True
# # #                     break

# # #         # If flagged, copy the image and its corresponding text label into separate error folders
# # #         if is_error:
# # #             dest_img_path = OUTPUT_IMAGE_DIR / img_path.name
# # #             dest_label_path = OUTPUT_LABEL_DIR / label_path.name
            
# # #             shutil.copy(img_path, dest_img_path)
# # #             if label_path.exists():
# # #                 shutil.copy(label_path, dest_label_path)
                
# # #             error_count += int(label_path.exists())

# # #     print(f"\n--- Filtering Complete ---")
# # #     print(f"Successfully isolated {error_count} error image-label pairs into '{OUTPUT_DIR}'.")

# # # if __name__ == "__main__":
# # #     run_clip_error_filtering()


# import os
# import shutil
# import cv2
# import torch
# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel

# # Setup device and model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_id = "openai/clip-vit-base-patch32"
# processor = CLIPProcessor.from_pretrained(model_id)
# model = CLIPModel.from_pretrained(model_id).to(device)

# def audit_yolo_dataset(images_dir, labels_dir, output_error_dir="images_error", darkness_threshold=45):
#     output_image_dir = os.path.join(output_error_dir, "images")
#     output_label_dir = os.path.join(output_error_dir, "labels")
#     os.makedirs(output_image_dir, exist_ok=True)
#     os.makedirs(output_label_dir, exist_ok=True)
    
#     candidate_labels = [
#         "a clear photo of a person", # Index 0: Baseline normal
#         "a photograph where the person is completely obscured by darkness",
#         "a photograph where the person is hidden by a strong shadow",
#         "a photograph where the person blends into the background and is difficult to see",
#         "a photograph where the person is partially invisible or visually indistinguishable from the background"
#     ]
    
#     flagged_count = 0
    
#     for label_file in os.listdir(labels_dir):
#         if not label_file.endswith(".txt"):
#             continue
            
#         base_name = os.path.splitext(label_file)[0]
#         image_path = None
#         image_ext = None
        
#         for ext in [".jpg", ".png", ".jpeg", ".JPG", ".JPEG", ".PNG"]:
#             potential_path = os.path.join(images_dir, base_name + ext)
#             if os.path.exists(potential_path):
#                 image_path = potential_path
#                 image_ext = ext
#                 break
                
#         if not image_path:
#             continue
            
#         image_cv = cv2.imread(image_path)
#         if image_cv is None:
#             continue
#         image_pil = Image.open(image_path).convert("RGB")
#         h, w, _ = image_cv.shape
        
#         label_path = os.path.join(labels_dir, label_file)
#         with open(label_path, "r") as f:
#             lines = f.readlines()
            
#         has_error = False
        
#         for line in lines:
#             parts = line.strip().split()
#             if not parts:
#                 continue
                
#             class_id = int(parts[0])
#             if class_id != 0:
#                 continue
                
#             x_center, y_center, box_w, box_h = map(float, parts[1:])
            
#             x_min = int((x_center - box_w / 2) * w)
#             y_min = int((y_center - box_h / 2) * h)
#             x_max = int((x_center + box_w / 2) * w)
#             y_max = int((y_center + box_h / 2) * h)
            
#             x_min, y_min = max(0, x_min), max(0, y_min)
#             x_max, y_max = min(w, x_max), min(h, y_max)
            
#             if x_max <= x_min or y_max <= y_min:
#                 continue
                
#             crop_cv = image_cv[y_min:y_max, x_min:x_max]
#             if crop_cv.size == 0:
#                 continue
#             crop_pil = image_pil.crop((x_min, y_min, x_max, y_max))
            
#             hsv_crop = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2HSV)
#             mean_brightness = hsv_crop[:, :, 2].mean()
#             is_excessively_dark = mean_brightness < darkness_threshold
            
#             inputs = processor(text=candidate_labels, images=crop_pil, padding=True, return_tensors="pt").to(device)
#             with torch.no_grad():
#                 outputs = model(**inputs)
            
#             probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
            
#             clear_score = probs[0]
#             max_error_score = probs[1:].max()
            
#             # STRICT CONDITION: Only flag if clear person score is very low and error score wins
#             if (clear_score < 0.35 and max_error_score > clear_score) or is_excessively_dark:
#                 has_error = True
#                 break
        
#         if has_error:
#             dest_image_path = os.path.join(output_image_dir, base_name + image_ext)
#             dest_label_path = os.path.join(output_label_dir, label_file)
            
#             shutil.copy(image_path, dest_image_path)
#             shutil.copy(label_path, dest_label_path)
#             flagged_count += 1

#     print(f"Audit complete! Strictly flagged {flagged_count} actual error images.")

# # Run the function:
# audit_yolo_dataset(
#     "/home/kasm-user/Desktop/folder/personmulti_final_base_hpo/images", 
#     "/home/kasm-user/Desktop/folder/personmulti_final_base_hpo/labels"
# )

# Code to delete extra labels with no matching image 
import os
from pathlib import Path

def clean_orphaned_labels(images_dir, labels_dir):
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    if not images_path.exists() or not labels_path.exists():
        print("[ERROR] Images or Labels directory does not exist!")
        return

    valid_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    
    # 1. Collect all image base names (stems) present in the images directory
    existing_image_stems = set()
    for img_path in images_path.glob("**/*"):
        if img_path.suffix in valid_extensions:
            existing_image_stems.add(img_path.stem)
            
    print(f"Found {len(existing_image_stems)} valid images in folder.")

    deleted_count = 0
    checked_count = 0

    # 2. Check all label files and delete any that don't have a matching image
    for label_file in labels_path.glob("*.txt"):
        checked_count += 1
        if label_file.stem not in existing_image_stems:
            # No corresponding image exists, so delete the orphaned label file
            label_file.unlink()
            deleted_count += 1

    print(f"\n--- Cleanup Complete ---")
    print(f"Checked {checked_count} label files.")
    print(f"Successfully deleted {deleted_count} orphaned label files.")

# Run the cleanup function with your dataset paths:
clean_orphaned_labels(
    "/home/kasm-user/Desktop/folder/personmulti_final_base_hpo/images", 
    "/home/kasm-user/Desktop/folder/personmulti_final_base_hpo/labels"
)
