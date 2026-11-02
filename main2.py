import os
import cv2
import csv
import numpy as np

# -----------------------------
# Paths
# -----------------------------
image_folder = "completely_visible/images/high_confidence"
label_folder = "completely_visible/labels/high_confidence"

output_folder = "completely_visible"
os.makedirs(output_folder, exist_ok=True)

csv_path = "person_bbox_analysis1.csv"


# -----------------------------
# Read image and label files
# -----------------------------
images = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

labels = sorted([
    f for f in os.listdir(label_folder)
    if f.lower().endswith(".txt")
])


# -----------------------------
# Colors
# -----------------------------
colors = {
    0: (0, 255, 0),     # person - green
    1: (255, 0, 0),     # hand - blue
    2: (0, 0, 255)      # head - red
}


# -----------------------------
# Statistics containers
# -----------------------------
bbox_widths = []
bbox_heights = []
bbox_areas = []

width_percents = []
height_percents = []
area_percents = []

aspect_ratios = []

frame_widths = []
frame_heights = []


# -----------------------------
# CSV creation
# -----------------------------
csv_file = open(csv_path, "w", newline="")

writer = csv.writer(csv_file)

writer.writerow([
    "Image",
    "Person_ID",
    "Frame_Width",
    "Frame_Height",
    "BBox_Width",
    "BBox_Height",
    "BBox_Area",
    "Width_Percentage",
    "Height_Percentage",
    "Area_Percentage",
    "Aspect_Ratio"
])


# -----------------------------
# Process images
# -----------------------------
for img_file, label_file in zip(images, labels):

    img_path = os.path.join(image_folder, img_file)
    label_path = os.path.join(label_folder, label_file)

    img = cv2.imread(img_path)

    if img is None:
        continue


    h, w = img.shape[:2]

    frame_widths.append(w)
    frame_heights.append(h)


    # -----------------------------
    # Display frame size
    # -----------------------------
    cv2.putText(
        img,
        f"Frame: {w}x{h}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    person_id = 0


    with open(label_path, "r") as f:

        for line in f:

            cls, xc, yc, bw, bh = map(float, line.split())

            cls = int(cls)


            # YOLO -> Pixel conversion
            box_w = int(bw * w)
            box_h = int(bh * h)

            x_center = int(xc * w)
            y_center = int(yc * h)


            x1 = max(0, int(x_center - box_w / 2))
            y1 = max(0, int(y_center - box_h / 2))

            x2 = min(w, int(x_center + box_w / 2))
            y2 = min(h, int(y_center + box_h / 2))


            # Draw all class boxes
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                colors.get(cls, (255,255,255)),
                2
            )


            # -----------------------------
            # Only analyse person class
            # -----------------------------
            if cls == 0:

                person_id += 1


                bbox_area = box_w * box_h
                frame_area = w * h


                width_pct = (box_w / w) * 100
                height_pct = (box_h / h) * 100
                area_pct = (bbox_area / frame_area) * 100


                aspect_ratio = (
                    box_w / box_h
                    if box_h > 0
                    else 0
                )

                # -----------------------------
                # Store statistics
                # -----------------------------
                bbox_widths.append(box_w)
                bbox_heights.append(box_h)
                bbox_areas.append(bbox_area)

                width_percents.append(width_pct)
                height_percents.append(height_pct)
                area_percents.append(area_pct)

                aspect_ratios.append(aspect_ratio)



                # -----------------------------
                # Save CSV row
                # -----------------------------
                writer.writerow([
                    img_file,
                    person_id,
                    w,
                    h,
                    box_w,
                    box_h,
                    bbox_area,
                    round(width_pct,2),
                    round(height_pct,2),
                    round(area_pct,2),
                    round(aspect_ratio,2),
                ])



                # =================================================
                # DISPLAY OPTIONS ON IMAGE
                # =================================================


                # -----------------------------
                # DEFAULT DISPLAY
                # Only bbox size
                # -----------------------------
                text1 = f"P{person_id}: {box_w}x{box_h}"

                cv2.putText(
                    img,
                    text1,
                    (x1, max(y1-10,30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    colors[0],
                    2
                )


                # -----------------------------
                # OPTIONAL:
                # Width and Height percentage
                # Uncomment below
                # -----------------------------

                # text2 = f"W:{width_pct:.1f}% H:{height_pct:.1f}%"

                # cv2.putText(
                #     img,
                #     text2,
                #     (x1, max(y1-30,50)),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.55,
                #     (255,255,0),
                #     2
                # )


                # -----------------------------
                # OPTIONAL:
                # Area percentage
                # -----------------------------

                # text3 = f"Area:{area_pct:.1f}%"

                # cv2.putText(
                #     img,
                #     text3,
                #     (x1, max(y1-50,70)),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.55,
                #     (255,255,0),
                #     2
                # )


                # -----------------------------
                # OPTIONAL:
                # Aspect ratio + size category
                # -----------------------------

                # text4 = f"AR:{aspect_ratio:.2f}"

                # cv2.putText(
                #     img,
                #     text4,
                #     (x1, y2+20),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.55,
                #     (0,255,255),
                #     2
                # )


                # -----------------------------
                # OPTIONAL:
                # Show everything
                # -----------------------------

                # info = [
                #     f"P{person_id}: {box_w}x{box_h}",
                #     f"W:{width_pct:.1f}% H:{height_pct:.1f}%",
                #     f"Area:{area_pct:.1f}%",
                #     f"AR:{aspect_ratio:.2f} {size}"
                # ]

                # for i, txt in enumerate(info):
                #     cv2.putText(
                #         img,
                #         txt,
                #         (x1, max(y1-(i+1)*20,20)),
                #         cv2.FONT_HERSHEY_SIMPLEX,
                #         0.5,
                #         (0,255,255),
                #         2
                #     )



    # Save annotated image
    cv2.imwrite(
        os.path.join(output_folder, img_file),
        img
    )


csv_file.close()



# -----------------------------
# Dataset Summary
# -----------------------------
print("\n")
print("="*70)
print("DATASET SUMMARY")
print("="*70)


if bbox_widths:

    print(f"Total Images Processed : {len(images)}")
    print(f"Total Persons          : {len(bbox_widths)}")


    print("\nFRAME SIZE")
    print(
        f"Width  -> Min:{min(frame_widths)} "
        f"Max:{max(frame_widths)}"
    )

    print(
        f"Height -> Min:{min(frame_heights)} "
        f"Max:{max(frame_heights)}"
    )


    print("\nBOUNDING BOX WIDTH")
    print(f"Min : {min(bbox_widths)}")
    print(f"Max : {max(bbox_widths)}")
    print(f"Mean: {np.mean(bbox_widths):.2f}")


    print("\nBOUNDING BOX HEIGHT")
    print(f"Min : {min(bbox_heights)}")
    print(f"Max : {max(bbox_heights)}")
    print(f"Mean: {np.mean(bbox_heights):.2f}")


    print("\nBOUNDING BOX AREA")
    print(f"Min : {min(bbox_areas)}")
    print(f"Max : {max(bbox_areas)}")
    print(f"Mean: {np.mean(bbox_areas):.2f}")


    print("\nAREA OCCUPANCY (%)")
    print(f"Min : {min(area_percents):.2f}")
    print(f"Max : {max(area_percents):.2f}")
    print(f"Mean: {np.mean(area_percents):.2f}")


    print("\nASPECT RATIO")
    print(f"Min : {min(aspect_ratios):.2f}")
    print(f"Max : {max(aspect_ratios):.2f}")
    print(f"Mean: {np.mean(aspect_ratios):.2f}")


print("\nAnnotated images saved in:", output_folder)
print("CSV saved in:", csv_path)