import numpy as np
import random
import cv2


def rotate_img(image, angle):
    # image: np.ndarray
    # angle: float
    #
    # grab the dim of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w//2, h//2)
    # grab the rotation matrix
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dim of the image
    nW = int((h*sin) + (w*cos))
    nH = int((h*cos) + (w*sin))

    M[0, 2] += (nW/2)-cX
    M[1, 2] += (nH/2)-cY

    image = cv2.warpAffine(image, M, (nW, nH))
    return image


def rotate_box(corners, angle, cx, cy, h, w):
    corners = corners.reshape(-1, 2)
    corners = np.hstack(
        (corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h*sin) + (w*cos))
    nH = int((h*cos) + (w*sin))

    M[0, 2] += (nW/2) - cx
    M[1, 2] += (nH/2) - cy

    calculated = np.dot(M, corners.T).T
    calculated = calculated.reshape(-1, 8)
    return calculated


def random_crop(image, poly_bboxes):
    """
    poly_bboxes has shape of (n,4,2) where n number of texts on the image (usually n=1).
    Each bounding box has 4 (x1,y1,x2,y2,x3,y3,x4,y4) points with 2 values.
    """
    (h, w) = image.shape[:2]
    poly_bboxes = poly_bboxes.reshape(-1, 2)
    xmax, ymax = poly_bboxes.max(axis=0)
    xmin, ymin = poly_bboxes.min(axis=0)
    xmin_crop_at = random.randint(0, xmin)
    xmax_crop_at = random.randint(xmax, w)

    ymin_crop_at = random.randint(0, ymin)
    ymax_crop_at = random.randint(ymax, w)

    cropped_image = image[ymin_crop_at:ymax_crop_at, xmin_crop_at:xmax_crop_at]
    new_corners = np.column_stack(
        (poly_bboxes[:, 0] - xmin_crop_at, poly_bboxes[:, 1] - ymin_crop_at))

    # reshape corners back to original shape
    return (cropped_image, new_corners.reshape(-1, 4, 2))
