import torch
torch.manual_seed(42)
import torch.nn as nn
import torchvision.models as models
import timm


# --- Backbone Model Loading Functions ---
# These functions load pre-trained models from torchvision or timm,
# remove their final classification layer, and return the feature extractor
# model along with the dimension of the output features.

def get_resnet50():
    """
    Loads a pre-trained ResNet50 model, removes its classification head,
    and returns the model and its feature dimension.

    Uses torchvision's recommended weights API.

    Returns:
        tuple: (torch.nn.Module, int) - The ResNet50 feature extractor and its output feature dimension (2048).
    """
    # Load a pretrained ResNet50 using the recommended weights
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Get the number of input features to the original fully connected layer
    num_features = model.fc.in_features  # typically 2048
    # Replace the final classification layer with an Identity layer to get features
    model.fc = nn.Identity()
    return model, num_features

def get_resnet101():
    """
    Loads a pre-trained ResNet101 model, removes its classification head,
    and returns the model and its feature dimension.

    Uses torchvision's recommended weights API.

    Returns:
        tuple: (torch.nn.Module, int) - The ResNet101 feature extractor and its output feature dimension (2048).
    """
    # Load a pretrained ResNet101 using the recommended weights
    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
    # Get the number of input features to the original fully connected layer
    num_features = model.fc.in_features # typically 2048
    # Replace the final classification layer with an Identity layer
    model.fc = nn.Identity()
    return model, num_features

def get_efficientnet_b3():
    """
    Loads a pre-trained EfficientNet-B3 model, removes its classification head,
    and returns the model and its feature dimension.

    Uses torchvision's recommended weights API.

    Returns:
        tuple: (torch.nn.Module, int) - The EfficientNet-B3 feature extractor and its output feature dimension (1536).
    """
    # Load a pretrained EfficientNet-B3 using the recommended weights
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    # EfficientNet's classifier is typically a Sequential(Dropout, Linear). Get Linear layer's in_features.
    num_features = model.classifier[1].in_features # typically 1536
    # Replace the entire classifier sequential block with an Identity layer
    model.classifier = nn.Identity()
    return model, num_features

def get_efficientnet_b7():
    """
    Loads a pre-trained EfficientNet-B7 model, removes its classification head,
    and returns the model and its feature dimension.

    Uses torchvision's recommended weights API.

    Returns:
        tuple: (torch.nn.Module, int) - The EfficientNet-B7 feature extractor and its output feature dimension (2560).
    """
    # Load a pretrained EfficientNet-B7 using the recommended weights
    model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
    # Get the input features of the final Linear layer in the classifier
    num_features = model.classifier[1].in_features # typically 2560
    # Replace the classifier block
    model.classifier = nn.Identity()
    return model, num_features

def get_mobilenet_v3():
    """
    Loads a pre-trained MobileNetV3-Large model, removes its classification head,
    and returns the model and its feature dimension.

    Uses torchvision's recommended weights API.

    Returns:
        tuple: (torch.nn.Module, int) - The MobileNetV3-Large feature extractor and its output feature dimension (960).
    """
    # Load a pretrained MobileNetV3-Large (Small version is commented out)
    # model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    # MobileNetV3's classifier is Sequential(Linear, Hardswish, Dropout, Linear). Get features before the first Linear layer.
    num_features = model.classifier[0].in_features # typically 960 for Large
    # Replace the classifier block
    model.classifier = nn.Identity()
    return model, num_features

def get_vgg16():
    """
    Loads a pre-trained VGG16 model, removes its classifier,
    and returns the model and the number of features before the original classifier.

    Note: VGG models output feature maps, not a single vector, before the classifier.
          The returned 'num_features' here corresponds to the input size of the *original*
          classifier's first layer, which depends on flattening the feature map.
          The actual output of the returned model will be a feature map (e.g., [batch, 512, 7, 7]).
          Further processing (like AdaptiveAvgPool2d) is needed to get a fixed-size vector.

    Uses torchvision's recommended weights API.

    Returns:
        tuple: (torch.nn.Module, int) - The VGG16 feature extractor (outputs feature map) and the input feature count of the original classifier (e.g., 25088).
    """
    # Load a pretrained VGG16
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    # Get the input feature count of the original classifier's first layer
    num_features = model.classifier[0].in_features  
    # Remove the entire classifier part to get features from the convolutional layers
    # model.classifier = nn.Identity() # This would keep the adaptive pooling if present
    model = nn.Sequential(*(list(model.children())[:-1])) # Removes the 'classifier' Sequential block entirely
    # The output of this model will be the feature map from the last conv block + pooling
    return model, num_features # Returning 4096 here might be misleading, as the model output isn't flat

def get_vgg19():
    """
    Loads a pre-trained VGG19 model, removes its classifier,
    and returns the model and the number of features before the original classifier.

    Similar note as get_vgg16 regarding output shape and num_features meaning.

    Uses torchvision's recommended weights API.

    Returns:
        tuple: (torch.nn.Module, int) - The VGG19 feature extractor (outputs feature map) and the input feature count of the original classifier (e.g., 25088).
    """
    # Load a pretrained VGG19
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    # Get the input feature count of the original classifier's first layer
    num_features = model.classifier[0].in_features # e.g., 4096 (after flattening 512*7*7 = 25088)
    # Remove the entire classifier part
    # model.classifier = nn.Identity()
    model = nn.Sequential(*(list(model.children())[:-1])) # Removes the 'classifier' Sequential block
    return model, num_features

def get_swin_transformer():
    """
    Loads a pre-trained Swin Transformer (Large version) model using timm,
    removes its classification head, and returns the model and its feature dimension.

    Returns:
        tuple: (torch.nn.Module, int) - The Swin Transformer feature extractor and its output feature dimension (1536 for Large).
    """
    # Load Swin-Large model from timm (Base version commented out)
    # model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
    model = timm.create_model('swin_large_patch4_window7_224.ms_in22k_ft_in1k', pretrained=True)
    # Get the feature dimension before classification head
    num_features = model.num_features  # For Swin-Large, this is 1536
    # Remove the classification head using timm's helper function
    model.reset_classifier(0) # 0 indicates removing the head
    return model, num_features

def get_deit():
    """
    Loads a pre-trained DeiT (Data-efficient Image Transformer - Base version) model using timm,
    removes its classification head, and returns the model and its feature dimension.

    Returns:
        tuple: (torch.nn.Module, int) - The DeiT feature extractor and its output feature dimension (768 for Base).
    """
    # Load DeiT-Base model from timm
    model = timm.create_model('deit_base_patch16_224', pretrained=True)
    # Get the input features to the classification head
    num_features = model.head.in_features # 768 for DeiT-Base
    # Replace the head with an Identity layer
    model.head = nn.Identity()
    return model, num_features

def get_vit():
    """
    Loads a pre-trained ViT (Vision Transformer - Base version) model using timm,
    removes its classification head, and returns the model and its feature dimension.

    Returns:
        tuple: (torch.nn.Module, int) - The ViT feature extractor and its output feature dimension (768 for Base).
    """
    # Load ViT-Base model from timm
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    # Get the input features to the classification head
    num_features = model.head.in_features # 768 for ViT-Base
    # Replace the head with an Identity layer
    model.head = nn.Identity()
    return model, num_features

# --- Model Dictionaries ---
# Dictionaries mapping model names (strings) to their corresponding loading functions.
# Allows easy selection of backbones by name.

# Dictionary for context models (can potentially differ from body/face)
context_backbone_dict = {
    'vgg16': get_vgg16,
    'vgg19': get_vgg19,
    'resnet50': get_resnet50,
    'resnet101': get_resnet101,
    'efficientnet_b3': get_efficientnet_b3,
    'efficientnet_b7': get_efficientnet_b7,
    'mobilenet_v3': get_mobilenet_v3,
    'swin': get_swin_transformer,
    'deit': get_deit,
    'vit': get_vit,
}

# Dictionary for body and face models (assuming they share the same pool of options)
body_face_dict = {
    'vgg16': get_vgg16,
    'vgg19': get_vgg19,
    'resnet50': get_resnet50,
    'resnet101': get_resnet101,
    'efficientnet_b3': get_efficientnet_b3,
    'efficientnet_b7': get_efficientnet_b7,
    'mobilenet_v3': get_mobilenet_v3,
    'swin': get_swin_transformer,
    'deit': get_deit,
    'vit': get_vit,
}

# --- Helper Functions for Model Retrieval ---

def get_context_model(model_name):
    """
    Retrieves a context feature extractor model and its feature dimension
    based on the provided model name using the context_backbone_dict.

    Args:
        model_name (str): The name of the context model (e.g., 'resnet50', 'swin').

    Returns:
        tuple: (torch.nn.Module, int) - The loaded feature extractor model and its feature dimension.

    Raises:
        ValueError: If the model_name is not found in context_backbone_dict.
    """
    # Check if the requested model name is a valid key in the dictionary
    if model_name not in context_backbone_dict:
        raise ValueError(f"Unsupported context model: {model_name}. Available: {list(context_backbone_dict.keys())}")
    # Call the corresponding function from the dictionary and return its result
    return context_backbone_dict[model_name]()

def get_body_model(model_name):
    """
    Retrieves a body feature extractor model and its feature dimension
    based on the provided model name using the body_face_dict.

    Args:
        model_name (str): The name of the body model (e.g., 'efficientnet_b7', 'deit').

    Returns:
        tuple: (torch.nn.Module, int) - The loaded feature extractor model and its feature dimension.

    Raises:
        ValueError: If the model_name is not found in body_face_dict.
    """
    # Check if the requested model name is valid
    if model_name not in body_face_dict:
        raise ValueError(f"Unsupported body model: {model_name}. Available: {list(body_face_dict.keys())}")
    # Call the function and return the model and feature dimension
    return body_face_dict[model_name]()

def get_face_model(model_name):
    """
    Retrieves a face feature extractor model and its feature dimension
    based on the provided model name using the body_face_dict.

    Args:
        model_name (str): The name of the face model (e.g., 'mobilenet_v3', 'vit').

    Returns:
        tuple: (torch.nn.Module, int) - The loaded feature extractor model and its feature dimension.

    Raises:
        ValueError: If the model_name is not found in body_face_dict.
    """
    # Check if the requested model name is valid
    if model_name not in body_face_dict:
        raise ValueError(f"Unsupported face model: {model_name}. Available: {list(body_face_dict.keys())}")
    # Call the function and return the model and feature dimension
    return body_face_dict[model_name]()
