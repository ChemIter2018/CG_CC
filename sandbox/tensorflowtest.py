import tensorflow as tf
from tensorflow import keras
import ssl

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

ssl._create_default_https_context = ssl._create_unverified_context

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test = keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')

# from tensorflow.python.compiler.mlcompute import mlcompute
# mlcompute.set_mlc_device(device_name="gpu")

with tf.device('/GPU:0'):
        model = keras.Sequential([keras.layers.Flatten(input_shape=(32,32,3)),
                                keras.layers.Dense(3000, activation='relu'),
                                keras.layers.Dense(1000, activation='relu'),
                                keras.layers.Dense(10, activation='sigmoid')
                        ])

        model.compile(optimizer="SGD", loss="categorical_crossentropy", metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5)
        model.evaluate(x_test, y_test, verbose=2)