import os
import tensorflow as tf
import numpy as np
import pandas as pd
from birdnet.models import BirdNET
from birdnet.preprocessing import AudioReader, MelSpectrogramPreprocessor, BatchPreprocessor


# Define parameters
batch_size = 32
num_epochs = 10
learning_rate = 1e-4
weight_decay = 1e-5

# Define paths
train_data_path = "path/to/train/data.csv"
val_data_path = "path/to/validation/data.csv"
checkpoint_dir = "path/to/checkpoint/directory"

# Load training and validation data
train_data = pd.read_csv(train_data_path)
val_data = pd.read_csv(val_data_path)

# Set up data preprocessing pipeline
audio_reader = AudioReader()
spectrogram_preprocessor = MelSpectrogramPreprocessor()
batch_preprocessor = BatchPreprocessor(audio_reader, spectrogram_preprocessor)

# Define model
model = BirdNET()
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metric_fn = tf.keras.metrics.BinaryAccuracy(threshold=0.5)

# Train the model
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    # Train on batches
    for batch_num, batch_data in enumerate(batch_preprocessor(train_data, batch_size)):
        batch_X, batch_y = batch_data
        with tf.GradientTape() as tape:
            logits = model(batch_X, training=True)
            loss_value = loss_fn(batch_y, logits) + weight_decay * tf.reduce_sum(model.losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        metric_fn.update_state(batch_y, logits)
        if batch_num % 10 == 0:
            print(f"Batch {batch_num}, Loss: {loss_value:.4f}, Accuracy: {metric_fn.result().numpy():.4f}")
    
    # Evaluate on validation data
    val_X, val_y = batch_preprocessor(val_data, len(val_data))
    val_logits = model(val_X, training=False)
    val_loss = loss_fn(val_y, val_logits)
    val_accuracy = metric_fn(val_y, val_logits)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy.numpy():.4f}")
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"ckpt_epoch{epoch+1}.h5")
    model.save_weights(checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
