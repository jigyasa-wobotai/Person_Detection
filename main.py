import os
import cv2

image_folder = "completely_visible/images/high_confidence"
label_folder = "completely_visible/labels/high_confidence"

output_folder = "annotated"
os.makedirs(output_folder, exist_ok=True)

class_names = {
    0: "person",
    1: "hand",
    2: "head"
}

colors = {
    0: (0, 255, 0),    # green person
    1: (255, 0, 0),    # blue hand
    2: (0, 0, 255)     # red head
}


images = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

labels = sorted([
    f for f in os.listdir(label_folder)
    if f.lower().endswith(".txt")
])


# First 50 images for testing
for img_file, label_file in zip(images[:50], labels[:50]):

    img_path = os.path.join(image_folder, img_file)
    label_path = os.path.join(label_folder, label_file)

    img = cv2.imread(img_path)

    if img is None:
        continue

    h, w = img.shape[:2]


    with open(label_path, "r") as f:

        for line in f:

            cls, xc, yc, bw, bh = map(float, line.split())

            cls = int(cls)

            # YOLO -> pixel coordinates
            box_w = int(bw * w)
            box_h = int(bh * h)

            x_center = int(xc * w)
            y_center = int(yc * h)

            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)

            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)


            # Draw rectangle
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                colors[cls],
                2
            )


            # Text
            text = f"{class_names[cls]} {box_w}x{box_h}"

            cv2.putText(
                img,
                text,
                (x1, max(y1-10,20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colors[cls],
                2
            )


    save_path = os.path.join(output_folder, img_file)

    cv2.imwrite(save_path, img)


print("Done. Annotated images saved in:", output_folder)