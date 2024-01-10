import cv2
from imutils.object_detection import non_max_suppression
import math
import numpy as np


def match(template, scaler, color):
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
    val_min, val_max, location_min, location_max = cv2.minMaxLoc(res)
    threshold = scaler * val_max  # Larger values have less, but better matches.
    (ys, xs) = np.where(res >= threshold)
    # Perform non-maximum suppression.
    h, w = template.shape[:2]
    rectangles = []
    for (x, y) in zip(xs, ys):
        rectangles.append((x, y, x + w, y + h))
    ipick = non_max_suppression(np.array(rectangles))
    # Optional: Visualize the results
    for (startX, startY, endX, endY) in ipick:
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    # cv2.imshow('Results', image)
    # cv2.waitKey(0)


def rotate(image1, angle, scale=1.0):
    h, w = image1.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angle, scale)

    rad = math.radians(angle)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    out_img = cv2.warpAffine(image1, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return out_img


template_image = cv2.imread('R_0805_10k_litest.png', 0)
template_image_180 = rotate(template_image, 180, scale=1)

templates = [
    template_image,
    template_image_180,
    rotate(template_image, 90, scale=1),
    rotate(template_image, 270, scale=1),
]

image = cv2.imread('board.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for _ in templates:
    match(_, 0.9, (0, 255, 0))

cv2.imshow('smd', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
