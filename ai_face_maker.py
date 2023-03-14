import os
import urllib.request
import tarfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# Download the CelebA dataset
url = "https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=1"
filename = "celeba.zip"
urllib.request.urlretrieve(url, filename)

# Extract the dataset
with tarfile.open(filename, "r") as tar:
    tar.extractall()
os.remove(filename)

# Define the path to the images directory
images_dir = "img_align_celeba"

# List all the image file names
image_files = os.listdir(images_dir)

# Print the total number of images
print("Total number of images:", len(image_files))


# Define the path to the images directory
images_dir = "img_align_celeba"

# Define the image size
img_size = 64

# Define the batch size for training
batch_size = 32

# Define the number of epochs for training
num_epochs = 100

# Preprocess the images
def preprocess_image(image_file):
    # Open the image using PIL
    image = Image.open(os.path.join(images_dir, image_file))
    # Crop the image to the face region
    image = image.crop((30, 30, 178, 178))
    # Resize the image to the desired size
    image = image.resize((img_size, img_size))
    # Convert the image to a numpy array
    image = np.array(image)
    # Normalize the pixel values to [-1, 1]
    image = (image / 255.0) * 2.0 - 1.0
    return image

# Create a dataset of preprocessed images
image_files = os.listdir(images_dir)
images = np.array([preprocess_image(file) for file in image_files])
dataset = tf.data.Dataset.from_tensor_slices(images)
dataset = dataset.shuffle(buffer_size=len(images)).batch(batch_size)

# Define the generator model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8 * 8 * 512, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((8, 8, 512)))
    assert model.output_shape == (None, 8, 8, 512)

    model.add(tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 256)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, img_size, img_size, 3)

    return model

# Define the discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[img_size, img_size, 3]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model


# Define the loss functions and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Define the GAN model
generator = make_generator_model()
discriminator = make_discriminator_model()

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Train the GAN model
for epoch in range(num_epochs):
    for batch in dataset:
        train_step(batch)

    # Generate a sample of artificial faces
    noise = tf.random.normal([batch_size, 100])
    generated_images = generator(noise, training=False)
    generated_images = (generated_images + 1.0) / 2.0
    generated_images = generated_images.numpy()

    # Postprocess the generated faces
    for i in range(len(generated_images)):
        generated_images[i] = np.clip(generated_images[i] * 255.0, 0.0, 255.0)
        generated_images[i] = generated_images[i].astype(np.uint8)

    # Output the generated faces to a file
    for i in range(len(generated_images)):
        plt.imsave(f"generated_faces/face_{epoch}_{i}.png", generated_images[i])
