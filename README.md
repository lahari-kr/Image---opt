# Image---opt
Image Generation Using GANs and Enhanced DCGAN

A deep learning project for generating synthetic geometric shapes using GAN architectures.

# Project Overview

This project focuses on generating synthetic geometric images (like circles) using two different Generative Adversarial Network (GAN) architectures:

1. Basic GAN (Baseline Model)

A simple and lightweight GAN architecture using:

Basic Conv2D / Conv2DTranspose layers

Standard Binary Cross Entropy

Standard GAN training loop

2. Enhanced Deep Convolutional GAN (Enhanced DCGAN)

An advanced and more stable GAN featuring:

9-layer deep generator with progressive refinement

6-layer discriminator with heavy regularization

Batch normalization, dropout, gradient clipping

Label smoothing & noise injection

3:1 discriminator-to-generator training ratio

This project compares the performance of both GANs on the same dataset of geometric shapes.

# Project Structure
Image-different-shapes/
│── dataset/
│   └── circles/        # Folder containing circle images (28x28 grayscale)
│
│── output_basic/       # Generated samples from Basic GAN
│── output_enhanced_dcgan/   # Generated samples from Enhanced DCGAN
│
│── shapes_gan.py       # Full GAN implementation (Basic + Enhanced DCGAN)
│── README.md           # Project documentation
└── requirements.txt     # Pip dependencies

 Implemented Models
🔹 Basic GAN

Simple architecture

Good for quick experimentation

Generator: 128 → 64 → 32 filters

Discriminator: 32 → 64 filters

🔹 Enhanced DCGAN (Advanced Model)

Deep generator (512 → 256 → 128 → 64 → 32 → 16 filters)

Deep discriminator with dropout & batch normalization

Gradient clipping for stability

Label smoothing

Noise-injected real images

Adaptive learning rates

 Dataset

The dataset must contain grayscale images of shapes such as circles:

dataset/circles/
    circle1.png
    circle2.png
    ...


Each image:

Size: 28×28

Color mode: grayscale

Normalized to [-1, 1]

# How to Run the Project
1. Install dependencies
pip install -r requirements.txt

2. Prepare Dataset

Place your circle images into:

dataset/circles/

3. Run Training

Inside shapes_gan.py, choose a model:

Train Basic GAN
gan = BasicGAN()
images = load_circles_dataset("dataset/circles")
gan.train(images, epochs=5000, batch_size=32)

Train Enhanced DCGAN
gan = EnhancedDCGAN()
images = load_circles_dataset("dataset/circles")
gan.train(images, epochs=5000, batch_size=32)

# Training Comparison Visualization

You can compare losses, accuracy, and smoothed curves using:

plot_training_comparison(basic_gan, enhanced_gan)


This generates:

Discriminator loss curves

Generator loss curves

Accuracy curves

Smoothed losses

Loss difference graphs

 Generated Samples

Both models save samples during training:

output_basic/circles_epoch_XXXX.png  
output_enhanced_dcgan/circles_epoch_XXXX.png


Enhanced DCGAN produces clearer and more realistic shape images.

 Features Included

✔️ Basic GAN implementation
✔️ Highly optimized Enhanced DCGAN
✔️ Noise and label smoothing
✔️ Gradient clipping
✔️ Training visualization
✔️ Save & load models
✔️ Generate single or multiple images

# Model Outputs

After training, models are saved as:

circle_generator_basic.h5
circle_generator_enhanced_dcgan.h5

 Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

# Future Improvements

Add more shape datasets (triangles, squares, mixed shapes)

Add WGAN-GP version

Improve dataset loader with augmentations
