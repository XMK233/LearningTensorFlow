from __future__ import absolute_import, division, print_function

import os
import platform

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

#data preparing
(train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

#model building method
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model

# load model saved by callback and weight only
checkpoint_path = "training_1/cpCW.h5"

model = create_model()
model.load_weights(checkpoint_path)

loss, acc = model.evaluate(test_images, test_labels)
print("Restored callback-weightOnly model, accuracy: {:5.2f}%".format(100*acc))

# load callback-entireModel
checkpoint_path = "training_1/cpCE.h5"

model = keras.models.load_model(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored callback-entire model, accuracy: {:5.2f}%".format(100*acc))

# manual saved model and weight only.
checkpoint_path = "training_1/cpMW.h5"

model = create_model()
model.load_weights(checkpoint_path)

loss, acc = model.evaluate(test_images, test_labels)
print("Restored manual-weightOnly model, accuracy: {:5.2f}%".format(100*acc))

#manual-entire model
checkpoint_path = "training_1/cpME.h5"

model = keras.models.load_model(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored manual-entire model, accuracy: {:5.2f}%".format(100*acc))