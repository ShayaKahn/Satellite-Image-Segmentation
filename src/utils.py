import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def get_image_mask_list(path):
    """
    Get the list of images and masks in the path.
    Inputs:
    -path: The path to the images and masks.
    Returns:
    - image_list: The list of image file names.
    - mask_list: The list of mask file names.
    """
    image_list = []
    mask_list = []
    for file_name in os.listdir(path):
        img_path = file_name
        if 'mask' in img_path:
            mask_list.append(img_path)
        elif 'sat' in img_path:
            image_list.append(img_path)
    return image_list, mask_list


def sort_image_mask_lists(image_list, mask_list):
    """
    Sort the image and mask lists.
    Inputs:
    - image_list: The list of image file names.
    - mask_list: The list of mask file names.
    Returns:
    - image_list_sorted: The sorted list of image file names.
    - mask_list_sorted: The sorted list of mask file names.
    """
    image_list_sorted = sorted(image_list)
    mask_list_sorted = sorted(mask_list)
    return image_list_sorted, mask_list_sorted

def display_image_mask(display_list):
    """
    Display an image, its ground true mask, and its predicted mask.
    Inputs:
    - display_list: A list of the image, ground true mask, and predicted mask.
    """
    plt.figure(figsize=(10, 10))

    title = ['Input Image', 'Ground true Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def create_confusion_matrix(real_mask_lst, pred_mask_lst, n_classes):
    """
    Create the confusion matrix.
    Inputs:
    - real_mask_lst: A list of real masks.
    - pred_mask_lst: A list of predicted masks.
    - n_classes: The number of classes.
    Returns:
    - cm: The confusion matrix.
    """
    # Initialize the confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)

    # Iterate over each pair of real and predicted masks
    for real_mask, pred_mask in zip(real_mask_lst, pred_mask_lst):
        # Compute the confusion matrix
        for true_class in range(n_classes):
            for predicted_class in range(n_classes):
                # Increment the confusion matrix for the true and predicted class combination
                cm[true_class, predicted_class] += np.sum(np.logical_and(
                    real_mask == true_class, pred_mask == predicted_class))
    return cm
