# Generative Adversarial Text-to-Image Synthesis

### Overview  
This project builds upon the foundational work of [aelnouby/Text-to-Image-Synthesis](https://github.com/aelnouby/Text-to-Image-Synthesis), which implements a GAN-based framework for synthesizing images from textual descriptions. While the original work focuses on integrating hierarchical text embeddings with GANs, this repository extends the approach by refining the architecture, optimizing training workflows, and incorporating custom dataset handling. The goal is to improve image quality, training stability, and scalability for text-to-image synthesis tasks.

---

## Key Differences from the Original Architecture  

### 1. **Architecture Modifications**  
The original repository uses a **StackGAN-like architecture** with two stages:  
- **Stage-I**: Generates low-resolution images conditioned on text embeddings.  
- **Stage-II**: Refines the output to higher resolution.  

**In this implementation**, the architecture is simplified and enhanced as follows:  
- **Single-Stage Training**: Focuses on generating **128x128 images** directly (compared to the original two-stage approach).  
- **Enhanced Encoder-Decoder Design**:  
  - **`SrcEncoder`**: A deeper convolutional network to extract richer visual features from source images.  
  - **`Generator`**: Combines encoded image features with **precomputed text embeddings** (see below) through concatenation and transposed convolutions.  
  - **`Discriminator`**: Uses a **patch-based adversarial loss** (outputs a 14x14 patch map) to improve local realism, unlike the original global discriminator.  
- **Simplified Conditioning**: Text embeddings are directly fused into the generator via spatial replication and concatenation, avoiding complex attention mechanisms.  

---

### 2. **Text Embedding Handling**  
- **Precomputed Embeddings**: Instead of generating text embeddings on-the-fly, this project uses **precomputed RNN-LSTM embeddings** from the [original author's repository](https://github.com/aelnouby/Text-to-Image-Synthesis). These embeddings are stored in HDF5 files and loaded during training.  
- **Padding and Normalization**: A custom collate function (`custom_collate_fn`) pads text embeddings to a fixed length of 328 dimensions (matching the original setup) and normalizes them for compatibility with the generator.  

---

### 3. **Dataset and Training Improvements**  
- **Class Splitting**: Explicit train/validation splits are defined (`train_classes` and `val_classes`) to prevent overfitting.  
- **Augmentation**: Uses `torchvision.transforms` for resizing, normalization, and data augmentation.  
- **Hyperparameters**:  
  - Batch size: **32**  
  - Learning rates: **5e-5 (Generator)**, **1e-4 (Discriminator)**  
  - Training epochs: **30**  
  - Image size: **128x128**  

---

## Setup and Usage  

### Dependencies  
To install the required packages, you can just run the first cell of the Jupyter notebook.

### Dataset Preparation  
1. Download and extract the dataset:  
   ```python
   !gdown 1VyNdHziY9avuv2_e3JG7LuW5hdkvv6hX  # From the repo
   !unzip -q data.zip
   ```
   or you can just run the afforementioned Jupyter notebook.
2. The dataset includes:  
   - Images in `./data/images`  
   - Precomputed text embeddings in `./data/text_c10` (from RNN-LSTM).  

---

## Results and Future Work  
- **Outputs**: Generated images are saved to `./output/models_{MODEL_NO}`.  
- **Visualization**: Use `TSNE` for embedding analysis and `matplotlib`/`plotly` for image grids.  

**Future Directions**:  
- Integrate attention mechanisms from the original work.  
- Experiment with larger resolutions (256x256).  
- Explore diffusion models for improved detail.  

---

## Acknowledgements  
- The text embeddings and baseline inspiration are derived from [aelnouby/Text-to-Image-Synthesis](https://github.com/aelnouby/Text-to-Image-Synthesis).  
- The Oxford-102 Flowers dataset is used for training.  