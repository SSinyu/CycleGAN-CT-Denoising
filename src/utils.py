import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.optimizers import schedules


def min_max_norm(img):
    out = ((img-img.min()) / (img.max()-img.min()))
    return out


def standardization(img):
    return (img-img.mean()) / img.std()


def percentile_thersholding(img, percentile=1):
    min_percentile = np.percentile(img, percentile)
    max_percentile = np.percentile(img, 100-percentile)

    img[img >= max_percentile] = max_percentile
    img[img <= min_percentile] = min_percentile
    return img


class ImageBuffer:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.items = []

    def __call__(self, in_items):
        if self.pool_size == 0:
            return in_items

        out_items = []
        for item in in_items:
            if len(self.items) < self.pool_size:
                self.items.append(item)
                out_items.append(item)
            else:
                if np.random.rand() < .5:
                    idx = np.random.randint(0, len(self.items))
                    out_item, self.items[idx] = self.items[idx], item
                    out_items.append(out_item)
                else:
                    out_items.append(item)
        return tf.stack(out_items, axis=0)


class LinearLRDecay(schedules.LearningRateSchedule):
    def __init__(self, init_lr, total_step, step_decay):
        super(LinearLRDecay, self).__init__()
        self.init_lr = init_lr
        self.total_steps = total_step
        self.step_decay = step_decay
        self.current_lr = tf.Variable(
            initial_value=init_lr,
            trainable=False,
            dtype=tf.float32
        )

    def __call__(self, step):
        self.current_lr.assign(
            tf.cond(
                step >= self.step_decay,
                true_fn = lambda: self.init_lr *
                        (1-1/(self.total_steps-self.step_decay) *
                        (step-self.step_decay)),
                false_fn = lambda: self.init_lr
            )
        )
        return self.current_lr
