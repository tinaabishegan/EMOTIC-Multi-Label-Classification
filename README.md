# Emotion Classification using Multi-Modal Frameworks on the EMOTIC Dataset

## 1. Overview

This repository contains the code and findings for a project focused on multi-label emotion classification using the EMOTIC dataset. Recognising human emotions in context is a complex task, influenced not only by facial expressions but also by body language and the surrounding environment. This project implements and empirically evaluates a multi-modal deep learning framework designed to leverage these different visual cues (context, body, and face) for improved emotion prediction accuracy.

The core contribution is a systematic, multi-stage empirical study that investigates the impact of various architectural components on classification performance. Specifically, we evaluate different:

* **Backbone Feature Extractors:** Comparing Convolutional Neural Networks (CNNs) and Vision Transformers.
* **Fusion Methods:** Exploring techniques to combine information from the different modalities.
* **Loss Functions:** Testing functions suitable for multi-label classification and addressing class imbalance.
* **Data Augmentation Strategies:** Assessing the effect of different image transformation techniques.
* **Dataset Resampling Techniques:** Employing ADASYN to mitigate the inherent class imbalance in the EMOTIC dataset.

The goal was to identify an optimal configuration of these components for the specific task of discrete emotion classification on EMOTIC. The primary evaluation metric used throughout the study is the mean Average Precision (mAP).

## 2. Dataset: EMOTIC

This project relies exclusively on the **EMOTIC (EMOTions In Context)** dataset.

* **Source & Citation:** The dataset was introduced by Kosti et al. (2017, 2019). If using this dataset or code, please cite their original work:
    * Kosti, R., Alvarez, J. M., Recasens, A., & Lapedriza, A. (2019). Context based emotion recognition using emotic dataset. *IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI)*. ([PDF via UOC](https://s3.sunai.uoc.edu/emotic/pdf/emotic_pami2019.pdf))
    * Kosti, R., Alvarez, J. M., Recasens, A., & Lapedriza, A. (2017). Emotion Recognition in Context. *CVPR*. ([PDF via UOC](https://s3.sunai.uoc.edu/emotic/pdf/emotic_cvpr17_paper.pdf))
* **Official Website:** [https://s3.sunai.uoc.edu/emotic/index.html](https://s3.sunai.uoc.edu/emotic/index.html)
* **Description:** EMOTIC comprises 23,571 images featuring 34,320 annotated people in diverse, unconstrained real-world settings ("in-the-wild"). The images are sourced from MSCOCO, Ade20k, and Google Search, ensuring variety in context and scenarios.
* **Annotations:**
    * **Bounding Boxes:** Provided for the person's body, face (where visible), and the full image (context).
    * **Discrete Emotions:** Each person is annotated with one or more labels from a set of **26 discrete emotion categories** (e.g., 'Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning'). This makes it a multi-label classification task.
    * **Continuous Dimensions:** Valence, Arousal, and Dominance (VAD) scores (1-10 scale) are also available but were not used in this project, which focused on discrete categories.
* **Challenges:** The dataset presents significant challenges, including:
    * **Class Imbalance:** Some emotion categories are far more frequent than others.
    * **Context Dependency:** Emotions are often subtle and highly dependent on the surrounding scene.
    * **Variability:** Wide variations in lighting, pose, occlusion, image quality, and background complexity.
* **Data Splits:** The project uses the standard train, validation, and test splits provided by the dataset creators. Annotation details for each split are contained within the `.csv` files (`train.csv`, `val.csv`, `test.csv`) located in the `emotic/emotic_pre/` directory.

## 3. Methodology

### 3.1. Multi-Modal Framework Architecture

The core architecture processes three visual streams independently before fusing their representations for final classification:

1.  **Context Stream:** Takes the entire image as input to capture environmental cues.
2.  **Body Stream:** Takes the cropped bounding box containing the person's body as input to capture pose and gesture information.
3.  **Face Stream:** Takes the cropped bounding box containing the person's face (if available) as input to capture facial expressions.

Each stream utilises a backbone feature extractor (pre-trained on ImageNet) to generate high-level feature vectors. These vectors are then passed to a fusion module, which combines them into a single representation. Finally, a classification head (typically a fully connected layer) predicts the probability for each of the 26 emotion categories.

### 3.2. Components Evaluated in Empirical Study

A multi-stage empirical study was conducted to systematically evaluate the impact of different components:

**a) Backbone Feature Extractors (`model.py`)**

Pre-trained models were used to leverage knowledge learned from large datasets like ImageNet. The following backbones were evaluated for each stream:

* **CNNs:**
    * `EfficientNet-B7`: State-of-the-art CNN known for efficiency and accuracy.
    * `MobileNetV3`: Lightweight CNN designed for mobile and resource-constrained environments.
* **Transformers:**
    * `Swin Transformer`: Hierarchical vision transformer using shifted windows, achieving strong performance.
    * `DeiT (Data-efficient Image Transformer)`: Vision transformer designed for effective training with less data than traditional ViTs.

**b) Fusion Methods (`fusion.py`)**

These modules combine the feature vectors from the context, body, and face streams. The following methods were tested:

* `Simple`: Concatenation of the feature vectors followed by a linear layer.
* `Weighted`: Simple concatenation, but the contribution of each stream is weighted by learnable parameters.
* `Attention`: Uses an attention mechanism to dynamically weight the importance of each stream's features.
* `Bilinear`: Captures pairwise interactions between features from different streams using bilinear pooling.
* `Bottleneck`: Reduces dimensionality of concatenated features through a bottleneck layer before expansion.
* `Cross-Modal Transformer`: Employs transformer layers to model interactions between features from different modalities.
* `Gated Residual`: Uses gating mechanisms to control information flow within a residual connection structure.
* `Hierarchical`: Fuses features in a step-wise manner (e.g., face+body first, then combined with context).
* `Layer Norm`: Applies Layer Normalization to the concatenated features.
* `Transformation`: Applies a learned linear transformation to the concatenated features.

**c) Loss Functions (`loss.py`)**

Given the multi-label nature and class imbalance of EMOTIC, several loss functions were evaluated:

* `BCE (Binary Cross-Entropy Loss)`: Standard loss for multi-label classification, treating each class independently.
* `Discrete Loss (Custom)`: A dynamically weighted version of BCE, potentially adjusting weights based on class frequency or other heuristics during training (implementation details in `loss.py`).
* `Focal Loss`: Modifies BCE to down-weight the loss contribution from easy-to-classify examples, focusing training on hard negatives. Effective for class imbalance.
* `ASL (Asymmetric Loss)`: Decouples the positive and negative contributions to the loss, allowing asymmetric focusing and probability shifting, particularly useful for imbalanced datasets.
* `Dice Loss`: Originally from segmentation, adapted here to maximise the overlap between predicted and true positive labels, potentially robust to imbalance.
* `MLLSC (Multi-Label Log-Sum-Exp Cosine Loss)`: A loss function designed to improve feature embedding for multi-label tasks.

**d) Image Augmentation (`augment.py`, `dataset.py`)**

Standard data augmentation techniques were applied during training to increase data diversity and model robustness. Three levels were tested:

* `none`: No augmentation applied.
* `basic`: Includes common augmentations like random horizontal flips, colour jitter (brightness, contrast, saturation), random affine transformations (rotation, translation, scaling), Gaussian blur, and Gaussian noise.
* `full`: Includes more aggressive or a wider range of augmentations (specifics defined in `augment.py`).

**e) Dataset Resampling (`train.py`, `utils.py`)**

To address the significant class imbalance, ADASYN (Adaptive Synthetic Sampling) was employed. ADASYN generates synthetic data samples for minority classes, focusing on harder-to-learn examples near class boundaries. It operates in the feature space extracted by the backbone models. Three levels were tested:

* `none`: No resampling applied.
* `mini`: ADASYN resampling.

### 3.3. Training Details

* **Optimiser:** Adam was used across all experiments.
* **Learning Rate:** A starting learning rate of 1e-5 was typically used, often with a scheduler (e.g., ReduceLROnPlateau).
* **Epochs:** Models were generally trained for 20 epochs, with early stopping based on validation mAP.
* **Batch Size:** Adjusted based on GPU memory constraints (e.g., 128).
* **Evaluation Metric:** Mean Average Precision (mAP) across the 26 categories was the primary metric for model selection and comparison.

## 4. Installation

1.  **Prerequisites:**
    * Python (tested with 3.10)
    * pip package manager
    * NVIDIA GPU with CUDA support (required for reasonable training times, tested with CUDA 11.8)
    * Git
2.  **Clone the Repository:**
    ```bash
    git clone tinaabishegan/EMOTIC-Multi-Label-Classification
    ```
3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 5. Usage

### 5.1. Dataset Setup

1.  **Download EMOTIC:** Obtain the dataset images and annotations from the [official EMOTIC website](https://s3.sunai.uoc.edu/emotic/index.html). You will need the image folders (e.g., `mscoco`, `framesdb`, `emodb_small`, `ade20k`) and the annotation CSV files.
2.  **Organise Files:** Place the downloaded image folders and the `train.csv`, `val.csv`, `test.csv` files into the `emotic/emotic_pre/` directory within the cloned project structure. The expected structure is:
    ```
    EmotionClassification/
    ├── emotic/
    │   ├── emotic_pre/
    │   │   ├── train.csv
    │   │   ├── val.csv
    │   │   ├── test.csv
    │   │   ├── mscoco/
    │   │   │   └── ... (image files)
    │   │   ├── framesdb/
    │   │   │   └── ... (image files)
    │   │   ├── emodb_small/
    │   │   │   └── ... (image files)
    │   │   └── ade20k/
    │   │       └── ... (image files)
    │   └── ...
    ├── results_stage1/
    ├── results_stage2/
    ├── ...
    ├── model.py
    ├── train.py
    ├── test.py
    ├── requirements.txt
    └── ... (other project files)
    ```
3.  **Preprocessing:** Image loading, resizing (to 224x224 or 256x256 depending on the backbone), cropping based on bounding boxes, and normalisation are handled automatically by the `Emotic_Dataset` class in `dataset.py`. Normalisation statistics depend on the backbone used (e.g., different stats for Swin vs. standard ImageNet stats).

### 5.2. Training a Model

* The main script for training is `train.py`.
* Hyperparameters and configuration options are passed via command-line arguments. Use `python train.py --help` to see all available options.
* Experiment configurations for each stage of the empirical study are also pre-defined in the `runner*.py` scripts (`runner1.py` for Stage 1, etc.). You can execute these directly (e.g., `python runner1.py`) or use them as a reference for `train.py` arguments.

**Example Training Command (Reflecting the Final Optimal Configuration):**

```bash
python train.py \
        --context_model "swin"
        --body_model "swin"
        --face_model "efficientnet_b7"
        --fusion "weighted"
        --ablation "all"
        --loss "discrete"
        --optimizer "adam"
        --epochs "20"
        --adasyn_level "mini"
        --context_aug_level "basic"
        --body_aug_level "basic"
        --face_aug_level "basic"
        --model_dir ./results/
```
context_model, body_model, face_model: Specify backbone models.

fusion: Specify the fusion method.

loss: Specify the loss function.

epochs: Maximum number of training epochs.

context_aug, body_aug, face_aug: Specify augmentation level ('none', 'basic', 'full').

adasyn_level: Specify ADASYN resampling level ('none', 'mini', 'full').

### 5.3. Evaluating a Trained Model

Use test.py to evaluate a saved model checkpoint on the test set.

Example Evaluation Command:
```bash
python test.py \
        --context_model "swin"
        --body_model "swin"
        --face_model "efficientnet_b7"
        --fusion "weighted"
        --results_dir ./results/
```

The script will output the mAP score and save detailed per-class AP scores to evaluation_metrics.txt in the specified results_dir.

## 6. Empirical Study Results

The study progressed through five distinct stages, iteratively refining the model configuration based on performance on the EMOTIC validation set. The components identified as optimal in each stage were carried forward to the next. The final performance was measured on the held-out test set using the best configuration found after Stage 5.

### Stage 1: Backbone Selection

* **Goal:** Identify the optimal combination of feature extractors for the context, body, and face streams.
* **Finding:** The combination of **Swin Transformer (Context) + Swin Transformer (Body) + EfficientNet-B7 (Face)** yielded the best performance. Transformers generally outperformed CNNs for context and body, likely due to their ability to capture global dependencies, while EfficientNet proved effective for the more localised face features.

### Stage 2: Fusion Method Selection

* **Goal:** Evaluate different fusion techniques using the best backbone combination from Stage 1.
* **Finding:** **Weighted Fusion** demonstrated the best performance. This suggests that allowing the model to learn the relative importance of each stream is beneficial.

### Stage 3: Loss Function Selection

* **Goal:** Determine the most effective loss function for the multi-label, imbalanced classification task.
* **Finding:** The custom **Discrete Loss** function consistently performed the best compared to other standard and imbalance-focused losses like BCE, Focal Loss, and ASL.

### Stage 4: Image Augmentation Evaluation

* **Goal:** Assess the impact of different levels of image augmentation.
* **Finding:** Applying **'basic' augmentation** (including standard flips, colour jitter, affine transforms, etc.) provided a noticeable improvement over no augmentation. More aggressive ('full') augmentation slightly degraded performance.

### Stage 5: Dataset Resampling Evaluation

* **Goal:** Investigate whether feature-space resampling using ADASYN could further improve performance by mitigating class imbalance.
* **Finding:** Applying **'mini' ADASYN** resulted in the highest performance boost observed in the study, confirming the importance of addressing class imbalance for this dataset.

### 6.1. Final Optimal Configuration and Test Performance

Based on the systematic evaluation across the five stages, the optimal configuration identified is:

* **Context Backbone:** Swin Transformer
* **Body Backbone:** Swin Transformer
* **Face Backbone:** EfficientNet-B7
* **Fusion Method:** Weighted Fusion
* **Loss Function:** Custom Discrete Loss
* **Image Augmentation:** Basic (applied to all streams)
* **Dataset Resampling:** ADASYN ('mini' level)
* **Optimiser:** Adam
* **Normalisation:** Swin-specific for Swin backbones, default ImageNet for EfficientNet.

**Final Test Set Performance:**

Using this optimal configuration, the model achieved a **mean Average Precision (mAP) of 32.00%** on the official EMOTIC test set.
