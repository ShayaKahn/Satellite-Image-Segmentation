from unittest import TestCase
import numpy as np
import tensorflow as tf
from src.utils import MeanIoU

class TestMeanIoU(TestCase):
    """
    This class tests the MeanIoU class.
    """
    def setUp(self) -> None:
        self.y_true = tf.convert_to_tensor(np.array([[[1, 0], [2, 0]], [[0, 1], [1, 2]]]), dtype=tf.int32)
        self.y_pred = tf.convert_to_tensor(np.array([[[[2.5, -1.2, -3.7], [1.1, -3.5, -4.1]],
                                                      [[-1.1, -1.2, 3.], [1.2, -1.9, -6.8]]],
                                                      [[[1.1, 0., -2.2], [9, 8.8, -3.3]],
                                                      [[0.1, 0.2, 0.05], [0.4, -1.2, -14.]]]]), dtype=tf.float32)
        self.mean_iou = MeanIoU(num_classes=3)
        self.mean_iou.update_state(self.y_true, self.y_pred)

    def test_update_state(self):
        """
        This method tests the update_state method.
        """
        self.assertAlmostEqual(self.mean_iou.result().numpy(), 0.4444, places=4)
