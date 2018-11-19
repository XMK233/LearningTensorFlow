from __future__ import absolute_import, division, print_function

import os
import platform

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

# data preparation
(train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# model building method
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

# models will be saved in this directory
#os.mkdir("training_1")

## using callback to save, only save weights. And then restore the weights
checkpoint_path = "training_1/cpCW.h5"
model = create_model()

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=0)

model.fit(train_images, train_labels,  epochs = 10,
          verbose=0,
          validation_data = (test_images,test_labels),
          callbacks = [cp_callback])  # pass callback to training

newModel = create_model()
newModel.load_weights(checkpoint_path)
loss, acc = newModel.evaluate(test_images, test_labels)
print("Restored callback-weightOnly model, accuracy: {:5.2f}%".format(100*acc))

# using callback to save, save entire model
checkpoint_path = "training_1/cpCE.h5"
model = create_model()

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=False,
                                                 verbose=0)

model.fit(train_images, train_labels,  epochs = 10,
          verbose=0,
          validation_data = (test_images,test_labels),
          callbacks = [cp_callback])  # pass callback to training

newModel = keras.models.load_model(checkpoint_path)
loss, acc = newModel.evaluate(test_images, test_labels)
print("Restored callback-entire model, accuracy: {:5.2f}%".format(100*acc))

# manually save a model. save only weights
checkpoint_path = "training_1/cpMW.h5"

model = create_model()
model.fit(train_images, train_labels,  epochs = 10,
          verbose=0,
          validation_data = (test_images,test_labels))
model.save_weights(checkpoint_path)

newModel = create_model()
newModel.load_weights(checkpoint_path)
loss, acc = newModel.evaluate(test_images, test_labels)
print("Restored manual-weightOnly model, accuracy: {:5.2f}%".format(100*acc))

# manually save entire model
checkpoint_path = "training_1/cpME.h5"

model = create_model()
model.fit(train_images, train_labels,  epochs = 10,
          verbose=0,
          validation_data = (test_images,test_labels))  # pass callback to training
model.save(checkpoint_path)

newModel = keras.models.load_model(checkpoint_path)
loss, acc = newModel.evaluate(test_images, test_labels)
print("Restored manual-entire model, accuracy: {:5.2f}%".format(100*acc))
