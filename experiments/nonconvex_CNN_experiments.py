# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 07:38:38 2022

@author: pccom
"""

import time
import os

# from tqdm.notebook import tqdm
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
import datetime

from frankwolfe.feasible_regions import create_lInf_constraints, make_feasible
from frankwolfe.application_specific_algorithms import AMSGrad,  SFW, AdaSFW, Adagrad, AdamSFW


def convert(x, y):
    x = tf.image.convert_image_dtype(x, tf.float32)
    x = (x - (0.4914, 0.4822, 0.4465)) / (0.247, 0.243, 0.261)
    return x, y


def augment(x, y):
    x = tf.image.resize_with_crop_or_pad(x, 36, 36)
    x = tf.image.random_crop(x, size=[32, 32, 3])
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_brightness(x, max_delta=1)
    return x, y


# function to reset metrics
def reset_metrics():
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()


# tf train function
@tf.function
def train(model, x, y):
    with tf.GradientTape() as tape:
        output = model(x, training=True)
        loss = loss_object(y, output)
    gradient = tape.gradient(loss, model.trainable_variables)
    return output, loss, gradient


# tf get_grad function
@tf.function
def get_grad(model, x, y):
    with tf.GradientTape() as tape:
        output = model(x, training=False)
        loss = loss_object(y, output)
    gradient = tape.gradient(loss, model.trainable_variables)
    return output, loss, gradient


# tf test function
@tf.function
def test(model, x, y):
    output = model(x, training=False)
    loss = loss_object(y, output)
    return output, loss


for optimizer_name in ["AdaSFW", "AdamSFW", "SFW", "SVRF", "ORGFW", "SPIDER-FW", "AMSGrad", "AdaGrad"]:
    
    assert optimizer_name in [
        "AdaSFW",
        "AdamSFW",
        "SFW",
        "MSFW",
        "SVRF",
        "ORGFW",
        "SPIDER-FW",
        "AMSGrad",
        "AdaGrad",
    ], "The optimizer chosen has not been implemented."
    
    reference_frequency = 0
    rho_base = 1
    rho_exponent = 0.6666666667
    
    if optimizer_name == "SFW":
        learning_rate_base = 0.03
        learning_rate_exponent = 0
        optimizer = SFW(learning_rate=learning_rate_base, momentum=0, rescale="diameter")
    if optimizer_name == "SVRF":
        learning_rate_base = 0.003
        learning_rate_exponent = 0
        reference_frequency = 1
        optimizer = SFW(learning_rate=learning_rate_base, momentum=0, rescale="diameter")
    if optimizer_name == "ORGFW":
        learning_rate_base = 0.3
        learning_rate_exponent = 0.6666666667
        rho_base = 1
        rho_exponent = 0.6666666667
        optimizer = SFW(learning_rate=learning_rate_base, momentum=0, rescale="diameter")
    if optimizer_name == "SPIDER-FW":
        learning_rate_base = 0.001
        learning_rate_exponent = 0
        reference_frequency = 1
        optimizer = SFW(learning_rate=learning_rate_base, momentum=0, rescale="diameter")
    if optimizer_name == "AdaSFW":
        learning_rate_base = 0.03
        learning_rate_exponent = 0
        inner_steps = 10
        optimizer = AdaSFW(learning_rate=learning_rate_base, inner_steps=inner_steps)
    if optimizer_name == "AdamSFW":
        learning_rate_base = 0.0003
        learning_rate_exponent = 0
        inner_steps = 5
        beta1 = 0.9
        beta2 = 0.999
        optimizer = AdamSFW(
            learning_rate=learning_rate_base,
            inner_steps=inner_steps,
            beta1=beta1,
            beta2=beta2,
        )
    if optimizer_name == "AdaGrad":
        learning_rate_base = 0.01
        learning_rate_exponent = 0
        optimizer = Adagrad(learning_rate=learning_rate_base, delta=1e-8)
    if optimizer_name == "AMSGrad":
        learning_rate_base = 0.0003
        learning_rate_exponent = 0
        beta1 = 0.9
        beta2 = 0.999
        optimizer = AMSGrad(
            learning_rate=learning_rate_base, delta=1e-8, beta1=beta1, beta2=beta2
        )
    
    
    # Define feasible region constraints for CIFAR-10 run.
    order = "inf"
    width = 100
    
    # Define some of the training parameters used in the training process.
    nepochs = 100
    batch_size = 100
    
    order = np.inf if order == "inf" else int(order)
    
    logs_per_epoch = 10  # @param {type:"integer"}
    include_baseline = True  # @param {type:"boolean"}
    verbose = True  # @param {type:"boolean"}
    
    
    dataset_name = "cifar10"
    model_type = "small CNN (used for CIFAR-10)"
    
    
    # Set code for CIFAR-10.
    input_shape = (32, 32, 3)
    nclasses = 10
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    
    
    # load dataset
    dataset, info = tfds.load(dataset_name, as_supervised=True, with_info=True)
    train_dataset, test_dataset = dataset["train"], dataset["test"]
    num_train_examples = info.splits["train"].num_examples
    num_test_examples = info.splits["test"].num_examples
    
    # pre-convert train dataset
    x_vals_train = np.zeros((num_train_examples,) + input_shape)
    y_vals_train = np.zeros(num_train_examples)
    for idx, (x, y) in tqdm(
        enumerate(train_dataset), total=num_train_examples, leave=False
    ):
        x_vals_train[idx], y_vals_train[idx] = convert(x, y)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_vals_train, y_vals_train))
    
    # pre-convert test dataset
    x_vals_test = np.zeros((num_test_examples,) + input_shape)
    y_vals_test = np.zeros(num_test_examples)
    for idx, (x, y) in tqdm(enumerate(test_dataset), total=num_test_examples, leave=False):
        x_vals_test[idx], y_vals_test[idx] = convert(x, y)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_vals_test, y_vals_test))
    
    # initialize model for CIFAR 10
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, kernel_size=(3, 3), input_shape=input_shape, activation="relu"
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    
    nlayers = len(model.trainable_variables)
    model.summary()
    # select constraints
    constraints = create_lInf_constraints(model, value=width, mode="initialization")
    
    # initialize pandas dataframe for storing metrics
    df_columns = [
        "epoch",
        "training loss",
        "test set loss",
        "training accuray",
        "test set accuracy",
        "gradients calculated",
        "time",
    ]
    df = pd.DataFrame(columns=df_columns)
    
    # reset model weights in case model was already trained
    model = tf.keras.models.clone_model(model)
    make_feasible(model, constraints)
    
    # define optimizer
    if optimizer_name in ["SFW", "MSFW", "SVRF", "ORGFW", "SPIDER-FW"]:
        optimizer = SFW(learning_rate=learning_rate_base, momentum=0, rescale="diameter")
    elif optimizer_name == "AdaSFW":
        optimizer = AdaSFW(learning_rate=learning_rate_base, inner_steps=inner_steps)
    elif optimizer_name == "AdamSFW":
        optimizer = AdamSFW(
            learning_rate=learning_rate_base,
            inner_steps=inner_steps,
            beta1=beta1,
            beta2=beta2,
        )
    elif optimizer_name == "AdaGrad":
        optimizer = Adagrad(learning_rate=learning_rate_base, delta=1e-8)
    elif optimizer_name == "AMSGrad":
        optimizer = AMSGrad(
            learning_rate=learning_rate_base, delta=1e-8, beta1=beta1, beta2=beta2
        )
    
    
    # define models needed vor variance reduction
    batch_model = tf.keras.models.clone_model(model)
    batch_model.set_weights(model.get_weights())
    vr_reference_model = tf.keras.models.clone_model(model)
    vr_reference_model.set_weights(model.get_weights())
    
    # define the loss object
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # initialize some necessary metrics objects
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy("train_accuracy")
    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy("test_accuracy")
    
    
    # baseline
    if include_baseline:
        for x, y in train_dataset.batch(batch_size):
            output, loss = test(model, x, y)
            train_loss.update_state(loss, sample_weight=len(y))
            train_accuracy.update_state(y, output, sample_weight=len(y))
    
        for x, y in test_dataset.batch(batch_size):
            output, loss = test(model, x, y)
            test_loss.update_state(loss, sample_weight=len(y))
            test_accuracy.update_state(y, output, sample_weight=len(y))
    
        if verbose:
            print("\nbaseline ---")
            print(
                f"train set loss {train_loss.result().numpy():.4f} and accuracy {train_accuracy.result().numpy()*100:.2f}%"
            )
            print(
                f"test set loss {test_loss.result().numpy():.4f} and accuracy {test_accuracy.result().numpy()*100:.2f}%"
            )
    
        new_row = pd.DataFrame(
            [
                [
                    0,
                    train_loss.result().numpy(),
                    test_loss.result().numpy(),
                    train_accuracy.result().numpy() * 100,
                    test_accuracy.result().numpy() * 100,
                    0,
                    time.process_time(),
                ]
            ],
            columns=df_columns,
        )
        df = df.append(new_row, ignore_index=True)
    
        reset_metrics()
    
    # determine parameters relevant for logging
    nbatches_per_epoch = math.ceil(num_train_examples / batch_size)
    nbatches_per_log = math.floor(nbatches_per_epoch / logs_per_epoch)
    log_batches = [
        nbatches_per_epoch - 1 - i * nbatches_per_log for i in range(logs_per_epoch)
    ]
    
    # determine parameters relevant for setting reference point
    if reference_frequency > 0:
        nbatches_per_reference = math.floor(nbatches_per_epoch / reference_frequency)
        reference_batches = [i * nbatches_per_reference for i in range(reference_frequency)]
    else:
        reference_batches = []
    
    # variables keeping track of the current batch number and the number of gradients calculated
    t = 0
    gradients_calculated = 0
    
    # training loop
    for epoch in range(1, nepochs + 1):
        # fix data augmentation throughout epoch to avoid clashing with variance reduction
        if dataset_name in ["cifar10", "cifar100"]:
            x_vals_augment = np.zeros((num_train_examples,) + input_shape)
            y_vals_augment = np.zeros(num_train_examples)
            for idx, (x, y) in tqdm(
                enumerate(train_dataset), total=num_train_examples, leave=False
            ):
                x_vals_augment[idx], y_vals_augment[idx] = augment(x, y)
            augmented_train_dataset = tf.data.Dataset.from_tensor_slices(
                (x_vals_augment, y_vals_augment)
            )
        else:
            augmented_train_dataset = train_dataset
    
        # train loop
        for batch_idx, (x, y) in enumerate(
            augmented_train_dataset.shuffle(num_train_examples).batch(batch_size)
        ):
            # update batch number, learning rate and momentum parameter
            t += 1
            learning_rate = learning_rate_base / t**learning_rate_exponent
            optimizer.set_learning_rate(learning_rate)
            rho = max(min(rho_base / t**rho_exponent, 1), 0)
    
            # update reference model and gradient
            if optimizer_name in ["SVRF", "SPIDER-FW"] and batch_idx in reference_batches:
                if verbose:
                    print(f"\n- setting new reference point for {optimizer_name} –")
                vr_reference_model.set_weights(model.get_weights())
                vr_reference_gradient = None
                for x, y in augmented_train_dataset.batch(batch_size):
                    output, loss, temp_gradient = get_grad(vr_reference_model, x, y)
                    gradients_calculated += len(y)
                    if vr_reference_gradient is None:
                        vr_reference_gradient = [
                            g * len(y) / num_train_examples for g in temp_gradient
                        ]
                    else:
                        vr_reference_gradient = [
                            rg + g * len(y) / num_train_examples
                            for rg, g in zip(vr_reference_gradient, temp_gradient)
                        ]
                    train_loss.update_state(loss, sample_weight=len(y))
                    train_accuracy.update_state(y, output, sample_weight=len(y))
    
            # get gradient from current batch and current model
            if not (batch_idx == 0 and optimizer_name in ["SVRF", "SPIDER-FW"]):
                output, loss, gradient = train(model, x, y)
                gradients_calculated += len(y)
                train_loss.update_state(loss, sample_weight=len(y))
                train_accuracy.update_state(y, output, sample_weight=len(y))
    
            # no gradient modification (outside of the optimizer class) for AdaSFW, AdamSFW and SFW
            if optimizer_name in ["AdaSFW", "AdamSFW", "SFW", "AdaGrad", "AMSGrad"]:
                modified_gradient = gradient
    
            # momentum gradient modification for MSFW
            elif optimizer_name == "MSFW":
                if batch_idx == 0 and epoch == 1:
                    momentum = [tf.zeros_like(w) for w in model.get_weights()]
                for idx, m in enumerate(momentum):
                    momentum[idx] = (1 - rho) * m + rho * gradient[idx]
                modified_gradient = momentum
    
            # unbiased momentum gradient modification for ORGFW
            elif optimizer_name == "ORGFW":
                if batch_idx == 0 and epoch == 1:
                    momentum = gradient
                else:
                    _, _, batch_gradient = get_grad(batch_model, x, y)
                    gradients_calculated += batch_size
                    for idx, m in enumerate(momentum):
                        momentum[idx] = (1 - rho) * (m - batch_gradient[idx]) + gradient[
                            idx
                        ]
                modified_gradient = momentum
    
            # variance reduction gradient modification for SVRF
            elif optimizer_name == "SVRF":
                if batch_idx in reference_batches:
                    modified_gradient = vr_reference_gradient
                else:
                    _, _, batch_gradient = get_grad(vr_reference_model, x, y)
                    gradients_calculated += batch_size
                    modified_gradient = [
                        vr_reference_gradient[i] + gradient[i] - batch_gradient[i]
                        for i in range(nlayers)
                    ]
    
            # variance reduction gradient modification for SPIDER
            elif optimizer_name == "SPIDER-FW":
                if batch_idx in reference_batches:
                    spider_accumulator = vr_reference_gradient
                else:
                    _, _, batch_gradient = get_grad(batch_model, x, y)
                    gradients_calculated += batch_size
                    for idx in range(len(spider_accumulator)):
                        spider_accumulator[idx] += gradient[idx] - batch_gradient[idx]
                modified_gradient = spider_accumulator
    
            # update batch_model
            batch_model.set_weights(model.get_weights())
    
            # update weights through optimizer
            optimizer.apply_gradients(
                zip(modified_gradient, model.trainable_variables, constraints)
            )
    
            # update log
            if batch_idx in log_batches:
    
                # test loop
                for x, y in test_dataset.batch(num_test_examples):
                    output, loss = test(model, x, y)
                    test_loss.update_state(loss, sample_weight=len(y))
                    test_accuracy.update_state(y, output, sample_weight=len(y))
    
                new_row = pd.DataFrame(
                    [
                        [
                            epoch - 1 + batch_idx / nbatches_per_epoch,
                            train_loss.result().numpy(),
                            test_loss.result().numpy(),
                            train_accuracy.result().numpy() * 100,
                            test_accuracy.result().numpy() * 100,
                            gradients_calculated,
                            time.process_time(),
                        ]
                    ],
                    columns=df_columns,
                )
                df = df.append(new_row, ignore_index=True)
    
                if verbose:
                    print(f"\nEPOCH {epoch-1+batch_idx/nbatches_per_epoch:.2f} ---")
                    print(
                        f"training loss {train_loss.result().numpy():.4f} and accuracy {train_accuracy.result().numpy()*100:.2f}%"
                    )
                    print(
                        f"test set loss {test_loss.result().numpy():.4f} and accuracy {test_accuracy.result().numpy()*100:.2f}%"
                    )
                    print(f"time elapsed {df['time'].iloc[-1] - df['time'].iloc[0]:.4f}s")
    
                reset_metrics()
    
    ts = time.time()
    timestamp = (
        datetime.datetime.fromtimestamp(ts)
        .strftime("%Y-%m-%d %H:%M:%S")
        .replace(" ", "-")
        .replace(":", "-")
    )
    
    output_directory = os.path.join(os.getcwd(), "output_data")
    # Make the directory if it does not already exist.
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        
    # Output the results as a pickled object for later use.
    filepath = os.path.join(
        os.getcwd(),
        "output_data",
        "CIFAR10_" + str(optimizer_name) + "_" + str(timestamp) + ".csv",
    )
    
    df.to_csv(filepath)
