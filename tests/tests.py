from unittest import TestCase
import numpy as np
import tensorflow as tf
from src.model import MeanIoU
from src.model import DiceLoss, JaccardLoss, CombinedLoss

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
        self.y_true = tf.one_hot(self.y_true, depth=3)
        self.mean_iou = MeanIoU(num_classes=3)
        self.mean_iou.update_state(self.y_true, self.y_pred)

    def test_update_state(self):
        """
        This method tests the update_state method.
        """
        self.assertAlmostEqual(self.mean_iou.result().numpy(), 0.4444, places=4)

class TestDiceLoss(TestCase):
    """
    This class tests the DiceLoss class.
    """
    def setUp(self) -> None:
        self.y_true = tf.convert_to_tensor(np.array([[[1, 0], [2, 0]]]), dtype=tf.int32)
        self.y_pred = tf.convert_to_tensor(np.array([[[[2.5, -1.2, -3.7], [1.1, -3.5, -4.1]],
                                                      [[-1.1, -1.2, 3.], [1.2, -1.9, -6.8]]]]), dtype=tf.float32)
        self.y_true = tf.one_hot(self.y_true, depth=3)
        self.dice_loss = DiceLoss(from_logits=True, smooth=0)
        self.loss = self.dice_loss.call(self.y_true, self.y_pred)

    def test_call(self):
        """
        This method tests the call method.
        """
        self.assertAlmostEqual(np.mean(self.loss.numpy()), 0.4, places=2)

class TestJaccardLoss(TestCase):
    """
    This class tests the JaccardLoss class.
    """
    def setUp(self) -> None:
        self.y_true = tf.convert_to_tensor(np.array([[[1, 0], [2, 0]]]), dtype=tf.int32)
        self.y_pred = tf.convert_to_tensor(np.array([[[[2.5, -1.2, -3.7], [1.1, -3.5, -4.1]],
                                                      [[-1.1, -1.2, 3.], [1.2, -1.9, -6.8]]]]), dtype=tf.float32)
        self.y_true = tf.one_hot(self.y_true, depth=3)
        self.jaccard_loss = JaccardLoss(from_logits=True, smooth=0)
        self.loss = self.jaccard_loss.call(self.y_true, self.y_pred)

    def test_call(self):
        """
        This method tests the call method.
        """
        print()
        self.assertAlmostEqual(np.mean(self.loss.numpy()), 0.4444, places=1)

class TestCombinedLoss(TestCase):
    """
    This class tests the CombinedLoss class.
    """
    def setUp(self) -> None:
        self.y_true = tf.convert_to_tensor(np.array([[[1, 0], [2, 0]]]), dtype=tf.int32)
        self.y_pred = tf.convert_to_tensor(np.array([[[[2.5, -1.2, -3.7], [1.1, -3.5, -4.1]],
                                                      [[-1.1, -1.2, 3.], [1.2, -1.9, -6.8]]]]), dtype=tf.float32)
        self.y_true = tf.one_hot(self.y_true, depth=3)
        self.jaccard_loss = JaccardLoss(from_logits=True, smooth=0)
        self.dice_loss = DiceLoss(from_logits=True, smooth=0)
        self.combined_loss = CombinedLoss(loss1=self.jaccard_loss, loss2=self.dice_loss)
        self.loss = self.combined_loss.call(self.y_true, self.y_pred)

    def test_call(self):
        """
        This method tests the call method.
        """
        comb_loss = (np.mean(self.dice_loss.call(self.y_true, self.y_pred).numpy()) + np.mean(
            self.jaccard_loss.call(self.y_true, self.y_pred).numpy())) / 2
        self.assertAlmostEqual(np.mean(self.loss.numpy()), comb_loss, places=2)