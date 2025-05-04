import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Default normalization parameters (non-Swin)
context_mean = [0.4690646, 0.4407227, 0.40508908]
context_std  = [0.2514227, 0.24312855, 0.24266963]
body_mean    = [0.43832874, 0.3964344, 0.3706214]
body_std     = [0.24784276, 0.23621225, 0.2323653]
face_mean    = [0.507395516207, 0.507395516207, 0.507395516207]
face_std     = [0.255128989415, 0.255128989415, 0.255128989415]

default_context_norm = [context_mean, context_std]
default_body_norm    = [body_mean, body_std]
default_face_norm    = [face_mean, face_std]

# SwinT-specific normalization (using ImageNet statistics)
swint_norm = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

class EmoticDataset(Dataset):
    """
    Custom Emotic dataset class.
    
    Args:
        x_context, x_body, x_face: Arrays of images (e.g. numpy arrays).
        y_cat, y_cont: Label arrays for categorical and continuous emotions.
        context_transform, body_transform, face_transform: Transformation pipelines.
            These are selected externally (for example, from augmentation dictionaries).
        context_norm, body_norm, face_norm: Normalization parameters (lists of [mean, std])
            to be used if the corresponding backbone is not Swin.
        isContextSwint (bool): If True, SwinT-specific normalization is applied to context images.
        isBodySwint (bool): If True, SwinT-specific normalization is applied to body images.
    
    Returns:
        A tuple of (transformed_context_image, transformed_body_image, transformed_face_image,
                   categorical_label, continuous_label_scaled)
    """
    def __init__(self, x_context, x_body, x_face, y_cat, y_cont, 
                 context_transform, body_transform, face_transform, 
                 context_norm, body_norm, face_norm, 
                 isContextSwint=False, isBodySwint=False):
        super(EmoticDataset, self).__init__()
        self.x_context = x_context
        self.x_body = x_body
        self.x_face = x_face
        self.y_cat = y_cat
        self.y_cont = y_cont

        # Use the provided transformation pipelines.
        self.context_transform = context_transform
        self.body_transform = body_transform
        self.face_transform = face_transform

        # Select normalization parameters based on whether the backbone is Swin.
        if isContextSwint:
            self.context_norm = transforms.Normalize(swint_norm[0], swint_norm[1])
        else:
            self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])
        
        if isBodySwint:
            self.body_norm = transforms.Normalize(swint_norm[0], swint_norm[1])
        else:
            self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])
        
        # Face always uses the default normalization.
        self.face_norm = transforms.Normalize(face_norm[0], face_norm[1])
        
    def __len__(self):
        return len(self.y_cont)
    
    def __getitem__(self, index):
        # Retrieve raw images and labels.
        image_context = self.x_context[index]
        image_body = self.x_body[index]
        image_face = self.x_face[index]
        cat_label = self.y_cat[index]
        cont_label = self.y_cont[index]
        
        # Apply transformation and then normalization.
        context_img = self.context_norm(self.context_transform(image_context))
        body_img = self.body_norm(self.body_transform(image_body))
        face_img = self.face_norm(self.face_transform(image_face))
        
        # Return transformed images and labels.
        # Note: Continuous labels are scaled (divided by 10.0) per original implementation.
        return (context_img,
                body_img,
                face_img,
                torch.tensor(cat_label, dtype=torch.float32),
                torch.tensor(cont_label, dtype=torch.float32) / 10.0)
