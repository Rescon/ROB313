import numpy as np
import cv2
from matplotlib import pyplot as plt


def gradient_orientation(img):
    """
    Calculate the gradient orientation for edge point in the image
    """
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    orientation = np.arctan2(grad_y, grad_x)
    orientation /= 2 * np.pi
    orientation += 0.5
    orientation = orientation.astype(np.float32)
    orientation = cv2.cvtColor(orientation, cv2.COLOR_GRAY2BGR)
    return orientation


def gradient_norme(img):
    """
    Calculate the gradient norme for edge point in the image
    """
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    norme = np.hypot(grad_x, grad_y)
    norme /= norme.max()
    return norme


def plot_orientation(frame, norme, orientation, orientation_mask):
    """
    Plot the gradient orientation for edge point in the image
    """
    fig = plt.figure(figsize=(12, 8))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.subplot(221)
    plt.imshow(frame)
    plt.title('Original')

    plt.subplot(222)
    plt.imshow(orientation)
    plt.title('Orientation de gradient')

    plt.subplot(223)
    plt.imshow(norme, cmap='gray')
    plt.title('Norme de gradient')

    plt.subplot(224)
    plt.imshow(orientation_mask)
    plt.title('Orientation')

    plt.show()


cap = cv2.VideoCapture('Test-Videos/Antoine_Mug.mp4')
th_min = 0.1  # threshold min

# cap = cv2.VideoCapture('Test-Videos/VOT-Ball.mp4')
# th_min = 0.06  # threshold min

# cap = cv2.VideoCapture('Test-Videos/VOT-Basket.mp4')
# th_min = 0.15  # threshold min

# cap = cv2.VideoCapture('Test-Videos/VOT-Car.mp4')
# th_min = 0.1  # threshold min

# cap = cv2.VideoCapture('Test-Videos/VOT-Sunshade.mp4')
# th_min = 0.05  # threshold min

# cap = cv2.VideoCapture('Test-Videos/VOT-Woman.mp4')
# th_min = 0.08  # threshold min


cpt = 1
while cpt < 30:
    ret, frame = cap.read()
    cpt += 1
cap.release()

frame_clone = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
orientation = gradient_orientation(frame_clone)
norme = gradient_norme(frame_clone)
orientation_mask = orientation.copy()
orientation_mask[np.where(norme < th_min)] = [1, 0, 0] # red

plot_orientation(frame, norme, orientation, orientation_mask)
