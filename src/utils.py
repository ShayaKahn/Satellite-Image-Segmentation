import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def get_image_mask_list(path):
    """
    Get the list of images and masks in the path.
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
    image_list_sorted = sorted(image_list)
    mask_list_sorted = sorted(mask_list)
    return image_list_sorted, mask_list_sorted

def display_image_mask(display_list):
    """
    Display an image, its ground true mask, and its predicted mask.
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

class MeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes, name=None, dtype=None):
        super(MeanIoU, self).__init__(num_classes=num_classes,
                                      name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert logits to predicted class labels
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)

        return super().update_state(y_true, y_pred, sample_weight)
