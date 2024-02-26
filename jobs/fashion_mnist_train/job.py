"""Example training job"""
# MNIST code heavily inspired by https://www.geeksforgeeks.org/fashion-mnist-with-python-keras-and-deep-learning/
import argparse
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import wandb
# To load the mnist data
from keras.datasets import fashion_mnist
# importing various types of hidden layers
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
# Adam legacy for m1/m2 macs
from tensorflow.keras.optimizers.legacy import Adam
from wandb.keras import WandbMetricsLogger


def train(project: Optional[str], entity: Optional[str], **kwargs: Any):
    run = wandb.init(project=project, entity=entity, config={
        "epochs": 10,
        "learning_rate": 0.001,
        "steps_per_epoch": 10,
    })

    # get config, could be set from sweep scheduler
    train_config = run.config
    # get training parameters from config
    epochs = train_config.get("epochs", 10)
    learning_rate = train_config.get("learning_rate", 0.001)
    steps_per_epoch = train_config.get("steps_per_epoch", 10)

    # load data
    (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()

    # Print the dimensions of the dataset
    print("Train: X = ", train_X.shape)
    print("Test: X = ", test_X.shape)

    # Lets cut this down for the sake of time
    train_X = train_X[:1200]
    train_y = train_y[:1200]

    # log some images
    for i in range(1, 10):
        # Create a 3x3 grid and place the
        # image in ith position of grid
        plt.subplot(3, 3, i)
        plt.imshow(train_X[i], cmap=plt.get_cmap("gray"))
    # Log plot
    wandb.log({"chart": plt})

    # Reshaping the arrays
    train_X = np.expand_dims(train_X, -1).astype(np.float32)
    test_X = np.expand_dims(test_X, -1)

    # load model
    model = model_arch()
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    model.summary()

    model.fit(
        train_X,
        train_y.astype(np.float32),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_split=0.33,
        callbacks=[
            WandbMetricsLogger(),
        ],
    )

    # do some manual testing

    labels = [
        "t_shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle_boots",
    ]
    # Make a prediction
    predictions = model.predict(test_X[:9])
    labels = [labels[np.argmax(pred)] for pred in predictions]

    for i in range(0, len(predictions)):
        # Create a 3x3 grid and place the
        # image in ith position of grid
        plt.subplot(3, 3, i + 1)
        plt.imshow(train_X[i], cmap=plt.get_cmap("gray"))
        plt.title(f"Pred: {labels[i]}")
    # Log plot
    wandb.log({"prediction-chart": plt})

# Code from: https://www.geeksforgeeks.org/fashion-mnist-with-python-keras-and-deep-learning/
def model_arch():
    """Define the architecture of the model"""
    models = Sequential()

    # We are learning 64
    # filters with a kernel size of 5x5
    models.add(
        Conv2D(64, (5, 5), padding="same", activation="relu", input_shape=(28, 28, 1))
    )

    # Max pooling will reduce the
    # size with a kernel size of 2x2
    models.add(MaxPooling2D(pool_size=(2, 2)))
    models.add(Conv2D(128, (5, 5), padding="same", activation="relu"))

    models.add(MaxPooling2D(pool_size=(2, 2)))
    models.add(Conv2D(256, (5, 5), padding="same", activation="relu"))

    models.add(MaxPooling2D(pool_size=(2, 2)))

    # Once the convolutional and pooling
    # operations are done the layer
    # is flattened and fully connected layers
    # are added
    models.add(Flatten())
    models.add(Dense(256, activation="relu"))

    # Finally as there are total 10
    # classes to be added a FCC layer of
    # 10 is created with a softmax activation
    # function
    models.add(Dense(10, activation="softmax"))
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", "-e", type=str, default=None)
    parser.add_argument("--project", "-p", type=str, default=None)
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
