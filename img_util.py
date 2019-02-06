import numpy as np
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
