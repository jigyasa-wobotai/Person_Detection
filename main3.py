import os
import cv2
import csv
import numpy as np


# ------------------------------------------------
# Paths
# ------------------------------------------------

image_folder = "completely_visible/images/high_confidence"
label_folder = "completely_visible/labels/high_confidence"

output_folder = "blur_analysis"

clear_folder = os.path.join(output_folder, "clear")
blurry_folder = os.path.join(output_folder, "blurry")
too_small_folder = os.path.join(output_folder, "too_small")

os.makedirs(clear_folder, exist_ok=True)
os.makedirs(blurry_folder, exist_ok=True)
os.makedirs(too_small_folder, exist_ok=True)


csv_path = os.path.join(
    output_folder,
    "blur_analysis_report.csv"
)


# ------------------------------------------------
# Blur threshold
# ------------------------------------------------
# Lower value = more sensitive to blur
# Adjust after checking dataset

BLUR_THRESHOLD = 50


# ------------------------------------------------
# Minimum bbox size
# ------------------------------------------------
# Blur estimation is unreliable for tiny objects

MIN_WIDTH = 32
MIN_HEIGHT = 32



# ------------------------------------------------
# Read files
# ------------------------------------------------

images = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg",".jpeg",".png"))
])


labels = sorted([
    f for f in os.listdir(label_folder)
    if f.lower().endswith(".txt")
])



# ------------------------------------------------
# CSV
# ------------------------------------------------

csv_file = open(
    csv_path,
    "w",
    newline=""
)

writer = csv.writer(csv_file)


writer.writerow([
    "Image",
    "Person_ID",
    "BBox_Width",
    "BBox_Height",
    "Laplacian_Variance",
    "Quality"
])



# ------------------------------------------------
# Statistics
# ------------------------------------------------

total_persons = 0
clear_count = 0
blurry_count = 0
too_small_count = 0

blur_values = []



# ------------------------------------------------
# Process images
# ------------------------------------------------

for img_file, label_file in zip(images, labels):


    img_path = os.path.join(
        image_folder,
        img_file
    )

    label_path = os.path.join(
        label_folder,
        label_file
    )


    img = cv2.imread(img_path)


    if img is None:
        continue


    h, w = img.shape[:2]


    image_status = "clear"


    person_id = 0



    with open(label_path, "r") as f:


        for line in f:


            cls, xc, yc, bw, bh = map(
                float,
                line.split()
            )


            cls = int(cls)


            # Only person class
            if cls != 0:
                continue



            person_id += 1
            total_persons += 1



            # --------------------------------
            # YOLO to pixel coordinates
            # --------------------------------

            box_w = int(bw * w)
            box_h = int(bh * h)


            x_center = int(xc * w)
            y_center = int(yc * h)


            x1 = max(
                0,
                int(x_center - box_w/2)
            )

            y1 = max(
                0,
                int(y_center - box_h/2)
            )


            x2 = min(
                w,
                int(x_center + box_w/2)
            )

            y2 = min(
                h,
                int(y_center + box_h/2)
            )



            roi = img[
                y1:y2,
                x1:x2
            ]



            # --------------------------------
            # Blur calculation
            # --------------------------------

            if box_w < MIN_WIDTH or box_h < MIN_HEIGHT:


                quality = "Too Small To Judge"

                color = (255,0,255)

                too_small_count += 1


                blur_score = 0



            else:


                gray = cv2.cvtColor(
                    roi,
                    cv2.COLOR_BGR2GRAY
                )


                blur_score = cv2.Laplacian(
                    gray,
                    cv2.CV_64F
                ).var()



                blur_values.append(
                    blur_score
                )



                if blur_score < BLUR_THRESHOLD:


                    quality = "Blurry"

                    color = (0,0,255)

                    image_status = "blurry"

                    blurry_count += 1


                else:


                    quality = "Clear"

                    color = (0,255,0)

                    clear_count += 1




            # --------------------------------
            # Draw bbox
            # --------------------------------

            cv2.rectangle(
                img,
                (x1,y1),
                (x2,y2),
                color,
                2
            )



            # --------------------------------
            # Display information
            # --------------------------------

            text = (
                f"P{person_id}: "
                f"{quality} "
                f"{blur_score:.1f}"
            )


            cv2.putText(
                img,
                text,
                (x1,max(y1-10,30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )



            # --------------------------------
            # Save CSV
            # --------------------------------

            writer.writerow([

                img_file,
                person_id,
                box_w,
                box_h,
                round(blur_score,2),
                quality

            ])




    # ------------------------------------------------
    # Save image based on final status
    # ------------------------------------------------

    if image_status == "blurry":


        cv2.imwrite(
            os.path.join(
                blurry_folder,
                img_file
            ),
            img
        )


    else:


        cv2.imwrite(
            os.path.join(
                clear_folder,
                img_file
            ),
            img
        )



csv_file.close()



# ------------------------------------------------
# Summary
# ------------------------------------------------

print("\n")
print("="*60)
print("BLUR ANALYSIS SUMMARY")
print("="*60)


print(
    "Total Persons:",
    total_persons
)

print(
    "Clear:",
    clear_count
)

print(
    "Blurry:",
    blurry_count
)

print(
    "Too Small To Judge:",
    too_small_count
)



if blur_values:

    print("\nLaplacian Variance")

    print(
        "Min:",
        round(min(blur_values),2)
    )

    print(
        "Max:",
        round(max(blur_values),2)
    )

    print(
        "Mean:",
        round(np.mean(blur_values),2)
    )



print("\nOutput saved:")
print(output_folder)

print("\nCSV:")
print(csv_path)