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


def preprocess(image, mask, size=256):
    # Resize the images and the masks
    input_image = tf.image.resize(image, (size, size), method='nearest')
    input_mask = tf.image.resize(mask, (size, size), method='nearest')

    return input_image, input_mask

def divide_tensor(image, mask):
    top_left = tf.slice(image, [0, 0, 0], [1224, 1224, 3])
    top_right = tf.slice(image, [0, 1224, 0], [1224, 1224, 3])
    bottom_left = tf.slice(image, [1224, 0, 0], [1224, 1224, 3])
    bottom_right = tf.slice(image, [1224, 1224, 0], [1224, 1224, 3])

    top_left_mask = tf.slice(mask, [0, 0, 0], [1224, 1224, 3])
    top_right_mask = tf.slice(mask, [0, 1224, 0], [1224, 1224, 3])
    bottom_left_mask = tf.slice(mask, [1224, 0, 0], [1224, 1224, 3])
    bottom_right_mask = tf.slice(mask, [1224, 1224, 0], [1224, 1224, 3])

    return [top_left, top_right, bottom_left, bottom_right], [top_left_mask, top_right_mask, bottom_left_mask, bottom_right_mask]

def expand_tensor(image, mask):
    subimages, submasks = divide_tensor(image, mask)

    return subimages, submasks

def preprocess_pipeline(image_list, mask_list, divide=False):
    dataset = create_shuffled_dataset(image_list, mask_list)
    image_ds = dataset.map(process_path)
    if divide:
        image_ds = image_ds.map(expand_tensor).unbatch()
    processed_image_ds = image_ds.map(preprocess)
    return processed_image_ds

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
    def __init__(self, seed=42, crop_height=96, crop_width=96, target_height=256, target_width=256):
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

def map_rgb_to_class(mask):
    """
    Converts RGB mask values to class indices.

    Parameters:
    - mask: A TensorFlow tensor of shape (height, width, 3).

    Returns:
    - A tensor of shape (height, width) with class indices.
    """
    # Convert mask to float32
    mask = tf.cast(mask, tf.float32)

    # Define color mappings (RGB to class index)
    color_to_class = {
        (0, 0, 0): 0,        # Black
        (0, 255, 0): 1,     # Green
        (255, 255, 0): 2,   # Yellow
        (0, 0, 255): 3,     # Blue
        (255, 0, 255): 4,   # Magenta
        (0, 255, 255): 5,   # Cyan
        (255, 255, 255): 6, # White
    }

    # Create an empty tensor to hold class indices with the same shape as the height and width of the mask
    class_indices = tf.zeros(shape=(tf.shape(mask)[0], tf.shape(mask)[1]), dtype=tf.uint8)

    # Apply each color mapping condition
    for color, class_index in color_to_class.items():
        # Create a boolean mask where all RGB channels match the given color
        color_condition = tf.reduce_all(tf.equal(mask, color), axis=-1)

        # Ensure the class_index is the same data type as class_indices
        class_index_tensor = tf.cast(class_index, class_indices.dtype)

        # set the class index at the locations where the condition is True
        class_indices = tf.where(color_condition, class_index_tensor, class_indices)

    return class_indices

def modify_mask(img, mask):
    """
    Converts the RGB mask into class indices using the map_rgb_to_class function.

    Parameters:
    - img: Image tensor of shape (height, width, 3).
    - mask: RGB mask tensor of shape (height, width, 3).

    Returns:
    - img: Unmodified image.
    - mask: Modified mask with class indices of shape (height, width).
    """
    class_mask = map_rgb_to_class(mask)

    return img, class_mask

def one_hot(img, mask):
    mask = tf.one_hot(mask, depth=7, axis=-1, dtype=tf.float32)
    return img, mask


#class Augment(tf.keras.layers.Layer):
#    def __init__(self, seed, crop_height, crop_width, target_height, target_width):
#        super().__init__()

        # Augmentation pipeline for images
#        self.augment_image = tf.keras.Sequential([
#            RandomFlip("horizontal_and_vertical", seed=seed),
#            RandomRotation(0.2, seed=seed),
#            RandomCrop(crop_height, crop_width, seed=seed),
#            tf.keras.layers.Resizing(target_height, target_width)
#        ])

        # Augmentation pipeline for masks (same augmentations to ensure consistency)
#        self.augment_mask = tf.keras.Sequential([
#            RandomFlip("horizontal_and_vertical", seed=seed),
#            RandomRotation(0.2, seed=seed),
#            RandomCrop(crop_height, crop_width, seed=seed),
#            tf.keras.layers.Resizing(target_height, target_width)
#        ])

#    def call(self, image, mask):
#        # Apply augmentations
#        image = self.augment_image(image)
#        image = tf.cast(image, dtype=tf.float32)

#        mask = self.augment_mask(mask)
#        mask = tf.cast(mask, dtype=tf.uint8)

#        return image, mask

# Encode the lables of the classes
#def eager_modify(element):
#    """
#    Reduce the channels of the mask to 1 by encoding the classes.
#    """
#    # transform to numpy array
#    element_numpy = element.numpy()
#    scaled_mask = element_numpy / 255.0
#    # Encode the classes
#    bool_mask = scaled_mask[:, :, 1] != 0
#    scaled_mask[:, :, 1][bool_mask] = 2
#    bool_mask = scaled_mask[:, :, 2] != 0
#    scaled_mask[:, :, 2][bool_mask] = 5
#    summed_mask = scaled_mask.sum(-1)
#    expanded_summed_mask = np.expand_dims(summed_mask, axis=-1).astype(np.uint8)
#    expanded_summed_mask[expanded_summed_mask == 2] = 1
#    expanded_summed_mask[expanded_summed_mask == 3] = 2
#    expanded_summed_mask[expanded_summed_mask == 5] = 3
#    expanded_summed_mask[expanded_summed_mask == 6] = 4
#    expanded_summed_mask[expanded_summed_mask == 7] = 5
#
#    return expanded_summed_mask

#def modify_mask(img, mask, size):
#    """
#    Modify the mask.
#    """
#    element = mask
#    modified_element = tf.py_function(func=eager_modify, inp=[element],
#                                      Tout=tf.uint8)

#    modified_element.set_shape((size, size, 1))

#    return img, modified_element

