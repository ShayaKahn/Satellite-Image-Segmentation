import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RandomRotation, RandomCrop, RandomFlip


def create_shuffled_dataset(image_list, mask_list):
    """
    Create a shuffled dataset from the image and mask lists.
    """
    # transform to numpy arrays
    image_list_array = np.array(image_list)
    mask_list_array = np.array(mask_list)

    # shuffle the arrays
    indices = np.random.permutation(len(image_list_array))

    image_list_array_shuffled = image_list_array[indices]
    mask_list_array_shuffled = mask_list_array[indices]

    image_list_array_shuffled = image_list_array_shuffled.tolist()
    mask_list_array_shuffled = mask_list_array_shuffled.tolist()

    # create a dataset
    image_filenames = tf.constant(image_list_array_shuffled)
    masks_filenames = tf.constant(mask_list_array_shuffled)

    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))

    return dataset


def process_path(image_path, mask_path):
    # process the image path
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # process the mask path
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)

    return img, mask


def preprocess(image, mask, size):
    # Resize the images and the masks
    input_image = tf.image.resize(image, (size, size), method='nearest')
    input_mask = tf.image.resize(mask, (size, size), method='nearest')

    return input_image, input_mask


def split(train_frac, val_frac, dataset):
    # Split the dataset into training, validation and test sets
    dataset_size = int(dataset.__len__())

    train_size = int(train_frac * dataset_size)
    val_size = int(val_frac * dataset_size)

    # Split the dataset into training, validation and test sets
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size)

    return train_dataset, val_dataset, test_dataset, train_size, val_size


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed, crop_height, crop_width, target_height, target_width):
        super().__init__()

        # Augmentation pipeline for images
        self.augment_image = tf.keras.Sequential([
            RandomFlip("horizontal_and_vertical", seed=seed),
            RandomRotation(0.2, seed=seed),
            RandomCrop(crop_height, crop_width, seed=seed),
            tf.keras.layers.Resizing(target_height, target_width)
        ])

        # Augmentation pipeline for masks (same augmentations to ensure consistency)
        self.augment_mask = tf.keras.Sequential([
            RandomFlip("horizontal_and_vertical", seed=seed),
            RandomRotation(0.2, seed=seed),
            RandomCrop(crop_height, crop_width, seed=seed),
            tf.keras.layers.Resizing(target_height, target_width)
        ])

    def call(self, image, mask):
        # Apply augmentations
        image = self.augment_image(image)
        image = tf.cast(image, dtype=tf.float32)

        mask = self.augment_mask(mask)
        mask = tf.cast(mask, dtype=tf.uint8)

        return image, mask

# Encode the lables of the classes
def eager_modify(element):
    """
    Reduce the channels of the mask to 1 by encoding the classes.
    """
    # transform to numpy array
    element_numpy = element.numpy()
    scaled_mask = element_numpy / 255.0
    # Encode the classes
    bool_mask = scaled_mask[:, :, 1] != 0
    scaled_mask[:, :, 1][bool_mask] = 2
    bool_mask = scaled_mask[:, :, 2] != 0
    scaled_mask[:, :, 2][bool_mask] = 5
    summed_mask = scaled_mask.sum(-1)
    expanded_summed_mask = np.expand_dims(summed_mask, axis=-1).astype(np.uint8)
    expanded_summed_mask[expanded_summed_mask == 2] = 1
    expanded_summed_mask[expanded_summed_mask == 3] = 2
    expanded_summed_mask[expanded_summed_mask == 5] = 3
    expanded_summed_mask[expanded_summed_mask == 6] = 4
    expanded_summed_mask[expanded_summed_mask == 7] = 5
    expanded_summed_mask[expanded_summed_mask == 8] = 6
    return expanded_summed_mask

def modify_mask(img, mask, size):
    """
    Modify the mask.
    """
    element = mask
    modified_element = tf.py_function(func=eager_modify, inp=[element],
                                      Tout=tf.uint8)

    modified_element.set_shape((size, size, 1))

    return img, modified_element

