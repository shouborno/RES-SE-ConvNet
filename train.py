from __future__ import absolute_import, division, print_function

import os
import shutil

import matplotlib.pyplot as plt
import numpy
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.initializers import Constant
from keras.layers import Add, Conv1D, GlobalAveragePooling1D, MaxPool1D, PReLU, Reshape, multiply
from loguru import logger
from sklearn.metrics import classification_report, cohen_kappa_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten, Input

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def get_sample_count(y):
    """
    Preparing dictionary for sample counts
    0: Normal (SpO2>91)
    1: Moderate (85>=SpoO2>=91)
    2: Severe (SpO2<85)
    """
    sample_counts = {0: 0, 1: 0, 2: 0}
    for label in y:
        if (isinstance(label, list) or isinstance(label, numpy.ndarray)) and len(label) == 1:
            label = label[0]
        elif not (isinstance(label, int) or isinstance(label, numpy.int64)):
            raise NotImplementedError(f"{type(label)} isn't supported")
        sample_counts[label] = sample_counts[label] + 1

    return sample_counts


def prepare_data():
    x = pd.read_csv(os.path.join(__location__, "data/ppg.csv"), header=None).values
    y = pd.read_csv(os.path.join(__location__, "data/Label.csv"), header=None).values
    logger.info(f"PPG dataset shape: {x.shape}")
    logger.info(f"Hypoxemia label shape: {y.shape}")

    sample_counts = get_sample_count(y)
    logger.info(f"Sample counts for each level of severity: {sample_counts}")

    train_x, test_x, train_y, test_y = train_test_split(
        x, y, train_size=0.7, test_size=0.3, random_state=122
    )

    oversampling_target = {1: 7000, 2: 7000}
    undersampling_target = {0: 7000}

    oversample = ADASYN(sampling_strategy=oversampling_target)
    x_over, y_over = oversample.fit_resample(x, y)

    undersample = RandomUnderSampler(sampling_strategy=undersampling_target)
    x_under, y_under = undersample.fit_resample(x_over, y_over)

    sample_counts = get_sample_count(y_under)
    logger.info(f"Sample counts after balancing: {sample_counts}")

    y_under = tf.keras.utils.to_categorical(y_under)
    y_under = y_under.reshape(y_under.shape[0], y_under.shape[1])
    test_y = tf.keras.utils.to_categorical(test_y)
    test_y = test_y.reshape(test_y.shape[0], test_y.shape[1])

    x_under = x_under.reshape(x_under.shape[0], x_under.shape[1], 1)
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], 1)

    return x_under, y_under, test_x, test_y


def SCNN(xin):
    r = Conv1D(64, kernel_size=3, padding="same", activation="relu")(xin)
    x = MaxPool1D(pool_size=2)(r)
    x = PReLU(alpha_initializer=Constant(value=0.2))(x)
    x = BatchNormalization()(r)
    return x


def Attention_SE(in_block, filters, ratio=16):
    x = GlobalAveragePooling1D()(in_block)
    x_shape = (1, 1, filters)
    x = Reshape(x_shape)(x)
    x = Dense(filters // ratio, activation="relu")(x)
    se = Dense(
        filters // ratio, activation="relu", kernel_initializer="he_normal", use_bias=False
    )(x)
    se = Dense(filters, activation="sigmoid", kernel_initializer="he_normal", use_bias=False)(se)

    x = multiply([in_block, se])

    return x


def RES_SE_ConvNet():
    x1_in = Input(shape=(125, 1))
    x1 = SCNN(x1_in)
    # Attestion_SE
    x_SE = Attention_SE(x1, 64, ratio=16)

    # parallel cnn branch

    x3 = SCNN(x1)
    x4 = SCNN(x3)
    x5 = SCNN(x4)
    x6 = SCNN(x5)

    x_c = Add()([x6, x_SE])

    c8 = Flatten()(x_c)
    x = Dense(256, activation="relu")(c8)
    x = Dense(128, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(3, activation="softmax")(x)

    m = Model(inputs=x1_in, outputs=x)

    m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    print(m.summary())
    return m


def train():
    model = RES_SE_ConvNet()

    x_train, y_train, x_test, y_test = prepare_data()

    batch_size = 256
    shutil.rmtree(os.path.join(__location__, "checkpoints"))
    checkpoint_path = os.path.join(__location__, "checkpoints/best.hdf5")

    checkpoint = ModelCheckpoint(
        checkpoint_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
    )
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.0001)
    callbacks_list = [checkpoint, reduce_lr]

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=150,
        callbacks=callbacks_list,
        validation_split=0.3,
    )

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    plt.figure()
    plt.plot(loss, "r", label="Training loss")
    plt.plot(val_loss, "b", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.ylim([0, 1])
    plt.xlim([0, 140])
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(val_accuracy, "r", label="Validation Accuracy")
    plt.plot(accuracy, "b", label="Training Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy Value")
    plt.ylim([0, 1])
    plt.xlim([0, 140])
    plt.legend()
    plt.show()

    y_pred = model.predict(x_test)

    y_test = numpy.argmax(y_test, axis=1)  # Convert one-hot to index
    y_pred = numpy.argmax(y_pred, axis=1)
    logger.info(classification_report(y_test, y_pred, digits=5))
    logger.info(f"Cohen Kappa Score: {cohen_kappa_score(y_test, y_pred)}")


if __name__ == "__main__":
    train()
