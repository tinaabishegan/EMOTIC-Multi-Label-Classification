import random
import numpy as np
from PIL import Image, ImageFilter # Pillow library for image manipulation
from torchvision import transforms # PyTorch transforms for image augmentation

# -------------------------
# Custom augmentation classes compatible with torchvision.transforms.Compose
# These classes apply augmentations probabilistically.
# -------------------------

class RandomGaussianBlur(object):
    """
    Apply Gaussian blur to a PIL Image with a specified probability
    using a randomly selected sigma value within a given range.
    """
    def __init__(self, kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5):
        """
        Initialises the RandomGaussianBlur transform.

        Args:
            kernel_size (tuple): Size of the Gaussian kernel (currently not used by PIL's GaussianBlur).
            sigma (tuple): A range (min, max) for the sigma (blur radius) value.
            p (float): The probability of applying the blur.
        """
        self.kernel_size = kernel_size # Note: PIL's GaussianBlur uses radius (sigma) not kernel size directly
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        """
        Applies the transform to the input image.

        Args:
            img (PIL.Image.Image): Input image.

        Returns:
            PIL.Image.Image: Blurred image or original image.
        """
        # Apply blur only if a random number is less than the probability p
        if random.random() < self.p:
            # Select a random sigma value from the specified range
            sigma_val = random.uniform(self.sigma[0], self.sigma[1])
            # Apply Gaussian blur using PIL's ImageFilter
            return img.filter(ImageFilter.GaussianBlur(radius=sigma_val))
        # Otherwise, return the original image
        return img

class RandomGaussianNoise(object):
    """
    Add random Gaussian noise to a PIL Image with a specified probability.
    """
    def __init__(self, mean=0.0, std=0.05, p=0.5):
        """
        Initialises the RandomGaussianNoise transform.

        Args:
            mean (float): Mean of the Gaussian distribution for noise.
            std (float): Standard deviation of the Gaussian distribution for noise.
            p (float): The probability of adding noise.
        """
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img):
        """
        Applies the transform to the input image.

        Args:
            img (PIL.Image.Image): Input image.

        Returns:
            PIL.Image.Image: Noisy image or original image.
        """
        # Apply noise only if a random number is less than the probability p
        if random.random() < self.p:
            # Convert PIL image to numpy array and normalise to [0, 1] float
            np_img = np.array(img).astype(np.float32) / 255.0
            # Generate Gaussian noise with the same shape as the image
            noise = np.random.normal(self.mean, self.std, np_img.shape)
            # Add noise to the image
            np_img = np_img + noise
            # Clip values to ensure they remain within the valid [0, 1] range
            np_img = np.clip(np_img, 0, 1)
            # Convert back to uint8 range [0, 255]
            np_img = (np_img * 255).astype(np.uint8)
            # Convert numpy array back to PIL image
            return Image.fromarray(np_img)
        # Otherwise, return the original image
        return img

class RandomCutMixV2(object):
    """
    Implements a variation of CutMix where a random rectangular patch
    in the image is replaced with a solid random colour. Applied with probability p.
    """
    def __init__(self, p=0.5):
        """
        Initialises the RandomCutMixV2 transform.

        Args:
            p (float): The probability of applying CutMix.
        """
        self.p = p

    def __call__(self, img):
        """
        Applies the transform to the input image.

        Args:
            img (PIL.Image.Image): Input image.

        Returns:
            PIL.Image.Image: Image with a patch replaced or original image.
        """
        # Apply CutMix only if a random number is less than the probability p
        if random.random() < self.p:
            # Get image dimensions
            w, h = img.size
            # Determine random width and height for the patch (10% to 40% of image dimensions)
            cut_w = int(w * random.uniform(0.1, 0.4))
            cut_h = int(h * random.uniform(0.1, 0.4))
            # Determine random top-left coordinates for the patch
            x0 = random.randint(0, w - cut_w)
            y0 = random.randint(0, h - cut_h)
            # Create a copy of the original image to modify
            img_copy = img.copy()
            # Generate a random RGB fill colour
            fill_color = tuple(np.random.randint(0, 256, size=3).tolist())
            # Create the patch as a solid colour image
            patch = Image.new('RGB', (cut_w, cut_h), fill_color)
            # Paste the patch onto the copied image
            img_copy.paste(patch, (x0, y0))
            # Return the modified image
            return img_copy
        # Otherwise, return the original image
        return img

class RandomMixUpV2(object):
    """
    Implements a variation of MixUp where the image is blended with its
    horizontally flipped version using a random alpha value. Applied with probability p.
    """
    def __init__(self, p=0.5):
        """
        Initialises the RandomMixUpV2 transform.

        Args:
            p (float): The probability of applying MixUp.
        """
        self.p = p

    def __call__(self, img):
        """
        Applies the transform to the input image.

        Args:
            img (PIL.Image.Image): Input image.

        Returns:
            PIL.Image.Image: Blended image or original image.
        """
        # Apply MixUp only if a random number is less than the probability p
        if random.random() < self.p:
            # Create a horizontally flipped version of the image
            img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
            # Choose a random blending factor (alpha) between 0.3 and 0.7
            alpha = random.uniform(0.3, 0.7)
            # Blend the original image and the flipped image using PIL's blend function
            # Result = img * (1.0 - alpha) + img_flip * alpha
            return Image.blend(img, img_flip, alpha)
        # Otherwise, return the original image
        return img

# -------------------------
# Composite transformations (using torchvision.transforms.Compose)
# These combine multiple standard and custom transforms.
# They are typically used for training and testing data pipelines.
# -------------------------

# --- Transforms for Non-Swin Transformer models ---

# Basic transform: Only converts to PIL Image and then to Tensor. No augmentation.
# Useful for validation/testing or when no augmentation is desired.
no_aug_transform = transforms.Compose([
    transforms.ToPILImage(), # Ensure input is PIL Image (if coming from numpy array)
    transforms.ToTensor()    # Convert PIL Image to PyTorch Tensor (scales to [0, 1])
])

# Basic augmentation: Horizontal flip and colour jittering.
basic_transform = transforms.Compose([
    transforms.ToPILImage(),             # Ensure input is PIL Image
    transforms.RandomHorizontalFlip(),   # Randomly flip horizontally (default p=0.5)
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # Adjust colour properties
    transforms.ToTensor()                # Convert to Tensor
])

# Full augmentation for context images: Includes geometric, colour, blur, noise, CutMix, MixUp.
full_context_transform = transforms.Compose([
    transforms.ToPILImage(),             # Ensure input is PIL Image
    transforms.RandomHorizontalFlip(),   # Horizontal flip
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), # Colour adjustments + hue
    transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.8, 1.2)), # Random rotation, translation, scaling
    RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5), # Custom Gaussian blur
    RandomGaussianNoise(mean=0.0, std=0.05, p=0.5),                 # Custom Gaussian noise
    RandomCutMixV2(p=0.5),                                          # Custom CutMix variation
    RandomMixUpV2(p=0.5),                                           # Custom MixUp variation
    transforms.ToTensor()                                           # Convert to Tensor
])

# Full augmentation for body images: Similar to context but without hue jitter, CutMix, MixUp.
full_body_transform = transforms.Compose([
    transforms.ToPILImage(),             # Ensure input is PIL Image
    transforms.RandomHorizontalFlip(),   # Horizontal flip
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # Colour adjustments
    transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.8, 1.2)), # Geometric transformations
    RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5), # Custom Gaussian blur
    RandomGaussianNoise(mean=0.0, std=0.05, p=0.5),                 # Custom Gaussian noise
    transforms.ToTensor()                                           # Convert to Tensor
])

# Full augmentation for face images: Uses the 'basic_transform' defined earlier.
# Less aggressive augmentation typically applied to faces.
full_face_transform = basic_transform

# -------------------------
# Swin Transformer specific transforms.
# Swin Transformers typically require a specific input size (e.g., 224x224).
# These transforms include resizing and center cropping before other augmentations.
# -------------------------

# Swin transform with no augmentation, only resize and crop.
swint_none_transform = transforms.Compose([
    transforms.ToPILImage(),             # Ensure input is PIL Image
    # Resize smaller edge to 232, maintaining aspect ratio, using BICUBIC interpolation
    transforms.Resize(size=[232], interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=[224]),   # Crop the center 224x224 patch
    transforms.ToTensor()                # Convert to Tensor
])

# Swin transform with basic augmentation (flip, colour jitter) after resize/crop.
swint_basic_transform = transforms.Compose([
    transforms.ToPILImage(),             # Ensure input is PIL Image
    transforms.Resize(size=[232], interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=[224]),
    transforms.RandomHorizontalFlip(),   # Horizontal flip
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # Colour adjustments
    transforms.ToTensor()                # Convert to Tensor
])

# Swin transform with full context augmentation after resize/crop.
swint_context_full_transform = transforms.Compose([
    transforms.ToPILImage(),             # Ensure input is PIL Image
    transforms.Resize(size=[232], interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=[224]),
    transforms.RandomHorizontalFlip(),   # Horizontal flip
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), # Colour adjustments + hue
    transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.8, 1.2)), # Geometric transformations
    RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5), # Custom Gaussian blur
    RandomGaussianNoise(mean=0.0, std=0.05, p=0.5),                 # Custom Gaussian noise
    RandomCutMixV2(p=0.5),                                          # Custom CutMix variation
    RandomMixUpV2(p=0.5),                                           # Custom MixUp variation
    transforms.ToTensor()                                           # Convert to Tensor
])

# Swin transform with full body augmentation after resize/crop.
# Note: This seems identical to swint_basic_transform in the provided code.
# It might be intended to have more augmentations like swint_context_full_transform but tailored for bodies.
swint_body_full_transform = transforms.Compose([
    transforms.ToPILImage(),             # Ensure input is PIL Image
    transforms.Resize(size=[232], interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=[224]),
    transforms.RandomHorizontalFlip(),   # Horizontal flip
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # Colour adjustments
    transforms.ToTensor()                # Convert to Tensor
])


# -------------------------
# Augmentation Level Dictionaries for selecting transforms easily.
# -------------------------

# Dictionaries mapping augmentation level names ("none", "basic", "full")
# to the corresponding transform compositions for different image types (context, body, face).

# For Default (Non-Swin) Transforms:
context_aug_levels = {
    "none": no_aug_transform,        # No augmentation
    "basic": basic_transform,       # Basic augmentation
    "full": full_context_transform  # Full context augmentation
}

body_aug_levels = {
    "none": no_aug_transform,        # No augmentation
    "basic": basic_transform,       # Basic augmentation
    "full": full_body_transform     # Full body augmentation
}

face_aug_levels = {
    "none": no_aug_transform,        # No augmentation
    "basic": basic_transform,       # Basic augmentation (used as 'full' for face)
    "full": full_face_transform     # Full face augmentation (same as basic)
}

# For Swin Transforms:
# Note: Swin transforms always include resize/crop, even at the "none" level.
swint_context_aug_levels = {
    "none": swint_none_transform,         # Resize/crop only
    "basic": swint_basic_transform,      # Resize/crop + basic aug
    "full": swint_context_full_transform # Resize/crop + full context aug
}

swint_body_aug_levels = {
    "none": swint_none_transform,         # Resize/crop only
    "basic": swint_basic_transform,      # Resize/crop + basic aug
    "full": swint_body_full_transform     # Resize/crop + full body aug (currently same as basic)
}

# Face transforms for Swin models are not explicitly defined here,
# likely reusing non-Swin face transforms or swint_basic_transform depending on the setup.

# -------------------------
# Data-level augmentation / Oversampling techniques
# These functions operate on feature vectors (X) and labels (y), typically
# after feature extraction or before the final classification layer,
# to address class imbalance in multi-label scenarios.
# They require scikit-learn for NearestNeighbors.
# -------------------------

def mini_batch_adasyn(X, y, k_neighbors=5):
    """
    Implements a simplified ADASYN (Adaptive Synthetic Sampling) approach
    designed to work on mini-batches of data for multi-label classification.
    It generates synthetic samples for minority classes within the batch,
    focusing on harder-to-learn instances (those near samples of other classes).
    This version includes corrections for neighbor selection during interpolation.

    Args:
        X (np.array): Feature matrix for the mini-batch (batch_size, n_features).
        y (np.array): Label matrix for the mini-batch (batch_size, n_classes), multi-label binary format.
        k_neighbors (int): Number of nearest neighbors to consider.

    Returns:
        tuple: (np.array, np.array) - Augmented feature matrix X_aug and label matrix y_aug for the batch.
               Returns original X, y if augmentation is not possible (e.g., batch too small, no minority class).
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        print("Error: scikit-learn is required for ADASYN/SMOTE. Please install it (`pip install scikit-learn`)")
        return X, y # Return original data if sklearn not available

    # Store original data, new samples will be appended
    X_aug_list = [X]
    y_aug_list = [y]
    n_samples, n_features = X.shape
    n_classes = y.shape[1]

    # --- Fit Nearest Neighbors on the entire batch ---
    # We need at least k_neighbors + 1 samples to find k neighbors.
    # k_actual_fit includes the point itself.
    k_actual_fit = min(k_neighbors + 1, n_samples)
    if k_actual_fit <= 1: # Need at least 2 samples (self + 1 neighbor)
        return X, y # Cannot find neighbors if batch size is too small

    # Initialise and fit the NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=k_actual_fit).fit(X)

    # --- Iterate through each class label ---
    for class_idx in range(n_classes):
        # Find indices of positive (minority) and negative (majority) samples for this class
        pos_idx = np.where(y[:, class_idx] == 1)[0]
        neg_idx = np.where(y[:, class_idx] == 0)[0]

        # --- Check if augmentation is needed/possible for this class ---
        # Skip if:
        # 1. No positive samples exist for this class in the batch.
        # 2. The positive class is not the minority class (or classes are balanced).
        if len(pos_idx) == 0 or len(pos_idx) >= len(neg_idx):
            continue

        # --- Calculate Synthesis Requirements ---
        # G: Total number of synthetic samples to generate for this class to balance the batch.
        G = len(neg_idx) - len(pos_idx)

        # --- Find Neighbors and Calculate Imbalance Ratio (r_i) ---
        # Find the k nearest neighbors for each positive sample within the batch.
        # `indices` shape: (n_positive_samples, k_actual_fit)
        # `distances` shape: (n_positive_samples, k_actual_fit)
        distances, indices = nbrs.kneighbors(X[pos_idx])

        r = [] # Stores the ratio of negative neighbors for each positive sample
        positive_neighbors_map = {} # Stores indices of *positive* neighbors for interpolation later

        # Iterate through each positive sample found
        for i_idx, i in enumerate(pos_idx): # i is the index in the original batch X
            # Get the indices of the neighbors for the current positive sample X[i]
            # Exclude the first neighbor, which is the point itself.
            neighbor_ids_in_batch = indices[i_idx][1:]

            # If no neighbors were found (can happen if k_actual_fit was 1 or duplicates exist)
            if len(neighbor_ids_in_batch) == 0:
                r.append(0) # No neighbors, so ratio is 0
                positive_neighbors_map[i] = [] # No positive neighbors either
                continue

            # Count how many of the neighbors belong to the negative class
            neg_count = np.sum(y[neighbor_ids_in_batch, class_idx] == 0)
            # Calculate the ratio r_i
            r.append(neg_count / len(neighbor_ids_in_batch))

            # Store the indices (in the original batch X) of neighbors that are *positive* for this class
            pos_neighbor_indices = [nid for nid in neighbor_ids_in_batch if y[nid, class_idx] == 1]
            positive_neighbors_map[i] = pos_neighbor_indices

        # Convert list of ratios to a numpy array
        r = np.array(r)

        # --- Calculate Density Distribution (d_i) ---
        # Normalize the ratios r_i to get a probability distribution d_i.
        # This determines how many synthetic samples are generated for each positive instance.
        # Instances with higher r_i (more negative neighbors) get higher d_i.
        if r.sum() == 0:
            # Handle edge case: If no positive samples have any negative neighbors (r_i are all 0).
            # This might mean the classes are well-separated in this batch.
            # Fallback: Assign uniform density if positive samples exist.
            if len(pos_idx) > 0:
                 d = np.ones_like(r) / len(pos_idx)
            else:
                 # Should not be reached due to the initial check, but for safety:
                 continue # Skip this class if no positive samples after all
        else:
            # Standard ADASYN: Normalize r_i
            d = r / r.sum()

        # --- Generate Synthetic Samples ---
        synthetic_samples = []
        synthetic_labels = []

        # Iterate through each positive sample again
        for i_idx, i in enumerate(pos_idx): # i is the index in the original batch X
            # G_i: Number of synthetic samples to generate for this specific positive instance X[i].
            G_i = int(np.round(d[i_idx] * G))

            # Skip if no samples are needed for this instance
            if G_i <= 0:
                continue

            # Get the indices of the k-nearest *positive* neighbors found earlier for X[i]
            current_pos_neighbors = positive_neighbors_map[i]

            # Cannot interpolate if the positive sample has no positive neighbors within the k set.
            if not current_pos_neighbors:
                continue

            # Generate G_i synthetic samples by interpolating with positive neighbors
            for _ in range(G_i):
                # **** CORRECTED NEIGHBOR SELECTION for Interpolation ****
                # ADASYN/SMOTE should interpolate between the minority sample and its *minority* neighbors.
                # Randomly choose one of the k-nearest *positive* neighbors.
                neighbor_idx = np.random.choice(current_pos_neighbors) # neighbor_idx is index in original batch X

                # Interpolate between the current sample X[i] and the chosen neighbor X[neighbor_idx]
                gap = np.random.rand() # Random value between 0 and 1
                synthetic = X[i] + gap * (X[neighbor_idx] - X[i])

                # Add the synthetic sample and its label (same as the original positive sample)
                synthetic_samples.append(synthetic)
                synthetic_labels.append(y[i]) # Assign the label of the original sample i

        # --- Collect augmented data for this class ---
        if synthetic_samples:
            # Convert lists to numpy arrays
            synthetic_samples = np.array(synthetic_samples)
            synthetic_labels = np.array(synthetic_labels)
            # Append to the overall lists for the batch
            X_aug_list.append(synthetic_samples)
            y_aug_list.append(synthetic_labels)

    # --- Combine original and all synthetic samples ---
    # Stack the original data and all generated synthetic samples vertically
    X_aug = np.vstack(X_aug_list)
    y_aug = np.vstack(y_aug_list)

    return X_aug, y_aug


def full_adasyn(X, y, k_neighbors=5):
    """
    Apply ADASYN (Adaptive Synthetic Sampling) to balance multi-label data across the entire dataset.
    Generates more synthetic samples for minority class instances that are
    harder to learn (those close to the decision boundary, i.e., having more neighbors
    from other classes). This is the standard ADASYN algorithm applied per class.

    Args:
        X (np.array): Feature matrix of shape (n_samples, n_features).
        y (np.array): Label matrix of shape (n_samples, n_classes) with binary labels (multi-label).
        k_neighbors (int): Number of nearest neighbors to use.

    Returns:
        tuple: (np.array, np.array) - Augmented feature matrix X_aug and label matrix y_aug.
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        print("Error: scikit-learn is required for ADASYN/SMOTE. Please install it (`pip install scikit-learn`)")
        return X, y

    # Start with the original data
    X_aug_list = [X]
    y_aug_list = [y]
    n_samples, n_features = X.shape
    n_classes = y.shape[1]

    # --- Iterate through each class label ---
    for class_idx in range(n_classes):
        # Find indices of positive (minority) and negative (majority) samples for this class
        pos_idx = np.where(y[:, class_idx] == 1)[0]
        neg_idx = np.where(y[:, class_idx] == 0)[0]

        # --- Check if augmentation is needed ---
        # Skip if the positive class is not the minority class for this label
        if len(pos_idx) == 0 or len(pos_idx) >= len(neg_idx):
            continue

        # --- Calculate Synthesis Requirements ---
        # G: Total number of synthetic samples needed for this class to balance it.
        G = len(neg_idx) - len(pos_idx)

        # --- Find Neighbors and Calculate Imbalance Ratio (r_i) ---
        # Fit NearestNeighbors on the entire dataset X
        # Ensure k is not larger than the number of samples - 1
        k_actual = min(k_neighbors, n_samples - 1)
        if k_actual < 1: # Need at least 1 neighbor
            continue
        # Find k_actual + 1 neighbors (including self)
        nbrs = NearestNeighbors(n_neighbors=k_actual + 1).fit(X)
        # Find neighbors specifically for the positive samples of the current class
        distances, indices = nbrs.kneighbors(X[pos_idx])

        # Calculate r_i: the ratio of *negative* samples among the k neighbors for each positive sample.
        r = np.zeros(len(pos_idx))
        for i, neighbors_for_i in enumerate(indices):
            # Exclude the first neighbor (which is the sample itself)
            neighbors_in_X = neighbors_for_i[1:]

            # Count how many of these neighbors have a *negative* label for the current class_idx
            if len(neighbors_in_X) > 0:
                neg_count = np.sum(y[neighbors_in_X, class_idx] == 0)
                r[i] = neg_count / len(neighbors_in_X)
            else:
                r[i] = 0 # No neighbors found (e.g., duplicates or k=0)

        # --- Calculate Density Distribution (d_i) ---
        # Normalize r_i to create a distribution d_i. Samples with more negative neighbors get higher probability.
        if r.sum() == 0:
            # If all r_i are 0 (e.g., classes perfectly separated), ADASYN cannot proceed adaptively.
            # Could potentially fallback to uniform distribution or SMOTE-like behavior,
            # but here we just skip augmentation for this class.
            print(f"Warning: ADASYN r_i sum is 0 for class {class_idx}. Skipping augmentation for this class.")
            continue

        d = r / r.sum()

        # --- Generate Synthetic Samples ---
        synthetic_samples = []
        synthetic_labels = []

        # Iterate through each positive sample for the current class
        for i, pos_sample_idx in enumerate(pos_idx): # pos_sample_idx is the index in the original X/y
            # G_i: Number of synthetic samples to generate based on this instance X[pos_sample_idx]
            G_i = int(np.round(d[i] * G))

            if G_i <= 0:
                continue

            # Get the neighbors (indices in X) for the current positive sample (excluding self)
            neighbors_in_X = indices[i, 1:]

            # Find which of these neighbors are also *positive* for the current class_idx
            # Interpolation should happen between samples of the *same* class (minority class).
            pos_neighbors_in_X = []
            for n_idx in neighbors_in_X:
                # Basic check for index validity (though shouldn't be necessary with sklearn's output)
                # Check if the neighbor n_idx also belongs to the positive class
                if n_idx < len(y) and y[n_idx, class_idx] == 1:
                     pos_neighbors_in_X.append(n_idx)

            # If there are no positive neighbors among the k neighbors, we cannot interpolate.
            if not pos_neighbors_in_X:
                # This could happen if a positive sample is isolated or surrounded only by negative samples.
                # Might need a strategy here, e.g., duplicate the sample, or skip. Current: skip.
                continue

            # Generate G_i synthetic samples
            for _ in range(G_i):
                # Randomly select one of the *positive* neighbors
                neighbor_idx = np.random.choice(pos_neighbors_in_X)

                # Generate the synthetic sample via linear interpolation
                gap = np.random.random() # Random float between 0.0 and 1.0
                synthetic_sample = X[pos_sample_idx] + gap * (X[neighbor_idx] - X[pos_sample_idx])

                # Append the new sample and its label (same as the original positive sample)
                synthetic_samples.append(synthetic_sample)
                synthetic_labels.append(y[pos_sample_idx]) # Label is copied from the base sample

        # --- Collect augmented data for this class ---
        if synthetic_samples:
            X_aug_list.append(np.array(synthetic_samples))
            y_aug_list.append(np.array(synthetic_labels))

    # --- Combine original and all synthetic samples ---
    X_aug = np.vstack(X_aug_list)
    y_aug = np.vstack(y_aug_list)

    return X_aug, y_aug


def smote(X, y, k_neighbors=5):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) for multi-label data, applied per class.
    Generates synthetic samples for minority classes by interpolating between
    existing minority samples and their k-nearest minority neighbors.
    Unlike ADASYN, SMOTE generates samples uniformly across all minority instances.

    Args:
        X (np.array): Feature matrix of shape (n_samples, n_features).
        y (np.array): Label matrix of shape (n_samples, n_classes) with binary labels (multi-label).
        k_neighbors (int): Number of nearest neighbors to use for synthetic sample generation.

    Returns:
        tuple: (np.array, np.array) - Augmented feature matrix X_aug and label matrix y_aug.
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        print("Error: scikit-learn is required for ADASYN/SMOTE. Please install it (`pip install scikit-learn`)")
        return X, y

    # Start with the original data
    X_aug_list = [X]
    y_aug_list = [y]
    n_samples, n_features = X.shape
    n_classes = y.shape[1]

    # --- Iterate through each class label ---
    for class_idx in range(n_classes):
        # Find indices of positive (minority) and negative (majority) samples for this class
        pos_idx = np.where(y[:, class_idx] == 1)[0]
        neg_idx = np.where(y[:, class_idx] == 0)[0]

        # --- Check if augmentation is needed ---
        # Skip if the positive class is not the minority class
        if len(pos_idx) == 0 or len(pos_idx) >= len(neg_idx):
            continue

        # --- Calculate Synthesis Requirements ---
        # n_synthetic: Total number of synthetic samples to generate for this class.
        n_synthetic = len(neg_idx) - len(pos_idx)

        # --- Find Neighbors (only among positive samples) ---
        # SMOTE finds neighbors only within the minority class samples.
        if len(pos_idx) > 1: # Need at least 2 positive samples to find neighbors
            # Ensure k is not larger than the number of positive samples - 1
            k_actual = min(k_neighbors, len(pos_idx) - 1)
            if k_actual < 1: # Need at least 1 neighbor
                continue # Cannot proceed if k=0

            # Fit NearestNeighbors *only* on the positive samples for this class
            nbrs = NearestNeighbors(n_neighbors=k_actual + 1).fit(X[pos_idx])
            # Find neighbors for each positive sample within the positive sample subset
            distances, indices_in_pos = nbrs.kneighbors(X[pos_idx])
            # indices_in_pos contains indices relative to the pos_idx array

            # --- Generate Synthetic Samples ---
            synthetic_samples = []
            synthetic_labels = []

            # Determine roughly how many samples to generate per positive instance
            # This aims for a uniform distribution of generated samples.
            samples_per_instance = int(np.ceil(n_synthetic / len(pos_idx)))

            generated_count = 0
            # Iterate through each positive sample (using its index within pos_idx)
            for i, original_pos_sample_idx in enumerate(pos_idx): # original_pos_sample_idx is index in X/y
                # Get the indices of the neighbors *within the pos_idx array* (excluding self)
                neighbor_indices_in_pos = indices_in_pos[i, 1:]

                # Generate samples for this instance, up to the total needed (n_synthetic)
                # Use min to avoid overshooting n_synthetic if samples_per_instance * len(pos_idx) > n_synthetic
                num_to_generate_for_i = min(samples_per_instance, n_synthetic - generated_count)

                for _ in range(num_to_generate_for_i):
                    # Randomly select one of the k nearest *positive* neighbors
                    # Choose an index from neighbor_indices_in_pos
                    chosen_neighbor_relative_idx = np.random.choice(neighbor_indices_in_pos)
                    # Convert the relative index back to the index in the original X/y array
                    chosen_neighbor_original_idx = pos_idx[chosen_neighbor_relative_idx]

                    # Interpolate between the current sample and the chosen neighbor
                    gap = np.random.random() # Random float 0.0 to 1.0
                    synthetic_sample = X[original_pos_sample_idx] + gap * (X[chosen_neighbor_original_idx] - X[original_pos_sample_idx])

                    # Append the new sample and its label
                    synthetic_samples.append(synthetic_sample)
                    # The label is the same as the original positive sample's label
                    synthetic_labels.append(y[original_pos_sample_idx])
                    generated_count += 1

                # Early exit if we have generated enough samples in total
                if generated_count >= n_synthetic:
                    break

            # --- Collect augmented data for this class ---
            if synthetic_samples:
                X_aug_list.append(np.array(synthetic_samples))
                y_aug_list.append(np.array(synthetic_labels))

    # --- Combine original and all synthetic samples ---
    X_aug = np.vstack(X_aug_list)
    y_aug = np.vstack(y_aug_list)

    return X_aug, y_aug
