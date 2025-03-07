# Deep Convolutional Generative Adversarial Network (DCGAN) Implementation

## Overview
This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) using PyTorch. DCGAN is an advanced architecture that combines Convolutional Neural Networks (CNNs) with the GAN framework to generate high-quality images.

## What is DCGAN?
DCGAN is a type of Generative Adversarial Network (GAN) that specifically uses convolutional and convolutional-transpose layers in the discriminator and generator, respectively. It was introduced to address the training stability issues of traditional GANs while producing higher quality images.

Key components:
1. **Generator**: Transforms random noise into synthetic images
2. **Discriminator**: Distinguishes between real and generated images

## Architecture Details

### Generator Architecture
- Takes random noise vector as input
- Uses transposed convolution layers for upsampling
- Batch normalization between layers
- ReLU activation functions
- Tanh activation for final layer
- Generates 64x64 RGB images

### Discriminator Architecture
- Convolutional layers for feature extraction
- Batch normalization
- LeakyReLU activation functions
- Binary classification output (real/fake)

## Implementation Details

Our implementation includes:
1. Data preprocessing and loading
2. Network architecture definition
3. Training loop with adversarial loss
4. Image generation and visualization

### Key Features
- Custom weight initialization
- Binary cross-entropy loss
- Adam optimizer
- Learning rate: 0.0002
- Beta1: 0.5 for Adam optimizer

## Dataset Details and Preprocessing
### Dataset
- Using the Celebrity Face Image Dataset from Kaggle
- Contains high-quality celebrity face images
- Images are processed to 64x64 RGB format

### Preprocessing Steps
1. **Image Resizing**: 
   - All images are resized to 64x64 pixels
   - Center cropping is applied to maintain aspect ratio

2. **Normalization**:
   - Images are converted to tensors
   - Normalized with mean (0.5, 0.5, 0.5) and std (0.5, 0.5, 0.5)
   - Pixel values scaled to [-1, 1] range

3. **Data Loading**:
   - Batch size: 128 images
   - Shuffled during loading
   - Uses 2 worker threads for data loading

## Training Instructions
1. **Environment Setup**:
   ```bash
   pip install torch torchvision torchaudio
   pip install kagglehub
```

## Training Process
The training process involves:
1. Training the discriminator on real and fake images
2. Training the generator to fool the discriminator
3. Alternating between these steps for multiple epochs
4. Monitoring loss values for both networks
5. Periodically generating sample images to track progress

## Results
The model generates synthetic images that progressively improve during training. The quality of generated images can be observed through:
- Generated sample images
- Loss curves for both generator and discriminator
- Visual comparison with real dataset images

## Dependencies
- PyTorch
- torchvision
- numpy
- matplotlib
- PIL

## Usage
1. Load the required dependencies
2. Prepare your dataset
3. Initialize the networks
4. Train the model
5. Generate new images using the trained generator

## Future Improvements
- Implement progressive growing of networks
- Add different loss functions
- Experiment with different architectures
- Try various hyperparameter combinations

## References
1. DCGAN Paper: "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
2. PyTorch Documentation
3. GAN Architecture Guidelines

## License
[Add your license information here]