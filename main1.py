import os
import cv2

image_folder = "partially-visible/images/high_confidence"
label_folder = "partially-visible/labels/high_confidence"

output_folder = "partially_visible"
os.makedirs(output_folder, exist_ok=True)


images = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

labels = sorted([
    f for f in os.listdir(label_folder)
    if f.lower().endswith(".txt")
])


colors = {
    0: (0, 255, 0),   # person - green
    1: (255, 0, 0),   # hand - blue
    2: (0, 0, 255)    # head - red
}


for img_file, label_file in zip(images, labels):

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


            # Draw all classes
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                colors[cls],
                2
            )


            # Show size ONLY for person class
            if cls == 0:

                text = f"person {box_w}x{box_h}"

                cv2.putText(
                    img,
                    text,
                    (x1, max(y1-10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    colors[cls],
                    2
                )


    cv2.imwrite(
        os.path.join(output_folder, img_file),
        img
    )


print("Done. Saved in:", output_folder)