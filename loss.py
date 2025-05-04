import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.nn.functional as F

class DiscreteLoss(nn.Module):
  ''' Class to measure loss between categorical emotion predictions and labels.'''
  def __init__(self, weight_type='mean', device=torch.device('cpu')):
    super(DiscreteLoss, self).__init__()
    self.weight_type = weight_type
    self.device = device
    if self.weight_type == 'mean':
      self.weights = torch.ones((1,26))/26.0
      self.weights = self.weights.to(self.device)
    elif self.weight_type == 'static':
      self.weights = torch.FloatTensor([0.1435, 0.1870, 0.1692, 0.1165, 0.1949, 0.1204, 0.1728, 0.1372, 0.1620,
         0.1540, 0.1987, 0.1057, 0.1482, 0.1192, 0.1590, 0.1929, 0.1158, 0.1907,
         0.1345, 0.1307, 0.1665, 0.1698, 0.1797, 0.1657, 0.1520, 0.1537]).unsqueeze(0)
      self.weights = self.weights.to(self.device)

  def forward(self, pred, target):
    if self.weight_type == 'dynamic':
      self.weights = self.prepare_dynamic_weights(target)
      self.weights = self.weights.to(self.device)
    loss = (((pred - target)**2) * self.weights)
    return loss.sum()

  def prepare_dynamic_weights(self, target):
    target_stats = torch.sum(target, dim=0).float().unsqueeze(dim=0).cpu()
    weights = torch.zeros((1,26))
    weights[target_stats != 0 ] = 1.0/torch.log(target_stats[target_stats != 0].data + 1.2)
    weights[target_stats == 0] = 0.0001
    return weights

class MultiLabelFocalLoss(nn.Module):
    """
    Multi-Label Focal Loss for addressing class imbalance in multi-label classification.
    
    The formulation modifies the binary cross-entropy loss by adding a modulating factor 
    (1 - p_t)^gamma and an optional weighting factor alpha. For each label:
    
        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    
    where:
      - p_t is the probability of the true class (after sigmoid),
      - gamma >= 0 focuses more on hard misclassified examples,
      - alpha (scalar or tensor of shape [num_classes]) balances the importance of positive/negative examples.

    Args:
        gamma (float): Focusing parameter. Default is 2.0.
        alpha (float or torch.Tensor, optional): Weighting factor for balancing positive and negative examples.
            If a float is provided, assumes same weight for all classes. If a tensor, should be of shape [num_classes].
            Default is None.
        reduction (str): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'. Default is 'mean'.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(MultiLabelFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Can be a float or a tensor of shape [num_classes]
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted logits of shape [batch_size, num_classes].
            targets (torch.Tensor): Ground truth binary labels of the same shape.
        
        Returns:
            torch.Tensor: Computed focal loss.
        """
        # Compute the sigmoid probability for each class
        p = torch.sigmoid(inputs)
        
        # Compute the binary cross entropy loss without reduction for numerical stability
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Get p_t: p for positive targets, (1-p) for negative targets
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Modulating factor (1 - p_t)^gamma
        modulating_factor = (1 - p_t) ** self.gamma

        # Compute the focal loss
        loss = modulating_factor * BCE_loss

        if self.alpha is not None:
            # If alpha is a scalar, create a weight tensor for positive & negative classes
            if isinstance(self.alpha, (float, int)):
                alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            else:
                # Assume alpha is a tensor of shape [num_classes]
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                # Broadcast alpha over the batch dimension
                alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            loss = alpha_factor * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification with imbalanced datasets.
    Applies different focusing parameters to positive and negative examples.
    """
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        # Sigmoid activation
        x_sigmoid = torch.sigmoid(x)
        
        # Clipping to prevent numerical instability
        if self.clip > 0:
            x_sigmoid = torch.clamp(x_sigmoid, min=self.clip, max=1.0-self.clip)
        
        # Calculate loss for positive and negative examples separately
        loss_pos = y * torch.log(x_sigmoid + self.eps)
        loss_neg = (1 - y) * torch.log(1 - x_sigmoid + self.eps)
        
        # Apply asymmetric focusing
        loss_pos = loss_pos * ((1 - x_sigmoid) ** self.gamma_pos)
        loss_neg = loss_neg * (x_sigmoid ** self.gamma_neg)
        
        return -torch.mean(loss_pos + loss_neg)

class MLLSCLoss(nn.Module):
    """
    Multi-Label Loss Correction (MLLSC) for handling missing and corrupted labels.
    Uses model confidence to identify and correct potentially wrong labels.
    """
    def __init__(self, tau=0.55, tau_prime=0.6):
        super(MLLSCLoss, self).__init__()
        self.tau = tau  # Threshold for positive label confidence
        self.tau_prime = tau_prime  # Threshold for negative label confidence
        
    def forward(self, x, y):
        probs = torch.sigmoid(x)
        
        # Indicators for high-confidence predictions
        positive_confident = (probs > self.tau).float()
        negative_confident = (probs < self.tau_prime).float()
        
        # Compute positive and negative loss terms with confidence-based correction
        L_pos = positive_confident * torch.log(probs + 1e-8) + \
                (1 - positive_confident) * torch.log(1 - probs + 1e-8)
        
        L_neg = negative_confident * torch.log(1 - probs + 1e-8) + \
                (1 - negative_confident) * torch.log(probs + 1e-8)
        
        # Combine losses based on ground truth
        loss = -torch.mean(y * L_pos + (1 - y) * L_neg)
        
        return loss

class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss for multi-label classification with class imbalance.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha  # Class weights (tensor of shape [num_classes])
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Compute class weights if not provided
        if self.alpha is None:
            # Calculate inverse class frequency as weights
            pos_count = torch.sum(targets, dim=0) + 1.0
            neg_count = targets.size(0) - pos_count + 1.0
            self.alpha = neg_count / (pos_count + neg_count)
            self.alpha = self.alpha.to(inputs.device)
        
        # Sigmoid activation
        probs = torch.sigmoid(inputs)
        
        # Calculate pt (probability of being correct)
        pt = probs * targets + (1 - probs) * (1 - targets)
        
        # Apply alpha weighting
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        alpha_factor = alpha_factor.to(inputs.device)
        
        # Apply focusing parameter
        focal_weight = (1 - pt) ** self.gamma
        
        # Compute binary cross entropy
        bce = -torch.log(pt + 1e-8)
        
        # Combine all terms
        loss = alpha_factor * focal_weight * bce
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class DiceLoss(nn.Module):
    """
    Dice Loss for multi-label classification, effective for imbalanced datasets.
    Optimizes the F1 score directly.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        # Apply sigmoid to inputs
        inputs = torch.sigmoid(inputs)
        
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and cardinalities
        intersection = (inputs * targets).sum()
        sum_inputs = inputs.sum()
        sum_targets = targets.sum()
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (sum_inputs + sum_targets + self.smooth)
        
        # Return Dice loss
        return 1.0 - dice