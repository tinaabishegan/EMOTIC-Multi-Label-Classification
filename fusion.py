import torch
import torch.nn as nn
import torch.nn.functional as F

# Global flag controlling prediction type.
# If isVADPrediction is False, the forward methods return categorical predictions;
# otherwise, they return continuous predictions.
isVADPrediction = False

def ensure_4d(x):
    """
    Ensures that the input tensor has 4 dimensions.
    If the input is 2D, unsqueeze spatial dimensions.
    """
    if x.dim() == 2:
        x = x.unsqueeze(-1).unsqueeze(-1)
    return x

class FusionCrossModalTransformer(nn.Module):
    """
    Fusion module using cross-modal transformer attention.
    
    For each modality (context, body, face), features are first projected into a
    common embedding space using 1x1 convolutions. The context features are adaptively
    pooled and flattened into tokens, which serve as the query for multihead attention,
    while the body and face tokens (concatenated) serve as keys and values.
    A residual connection and feed-forward network further refine the tokens,
    and their average pooled output is used for final prediction.
    """
    def __init__(self, num_context_features, num_body_features, num_face_features,
                 embed_dim=256, num_heads=8, num_classes=26, num_cont=3):
        super(FusionCrossModalTransformer, self).__init__()
        self.embed_dim = embed_dim
        
        # Projection layers for each modality.
        self.proj_context = nn.Conv2d(num_context_features, embed_dim, kernel_size=1)
        self.proj_body    = nn.Conv2d(num_body_features, embed_dim, kernel_size=1)
        self.proj_face    = nn.Conv2d(num_face_features, embed_dim, kernel_size=1)
        
        # Multihead attention (using context tokens as query).
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        # Final classifier heads.
        self.fc_cat = nn.Linear(embed_dim, num_classes)
        self.fc_cont = nn.Linear(embed_dim, num_cont)
    
    def forward(self, x_context, x_body, x_face):
        B = x_context.size(0)
        # Ensure inputs are 4D.
        x_context = ensure_4d(x_context)
        x_body = ensure_4d(x_body)
        x_face = ensure_4d(x_face)
        
        # Project each modality.
        proj_context = self.proj_context(x_context)   # (B, embed_dim, H, W)
        proj_body    = self.proj_body(x_body)
        proj_face    = self.proj_face(x_face)
        
        # Adaptive pooling to a fixed spatial size (7x7).
        target_size = (7, 7)
        proj_context = F.adaptive_avg_pool2d(proj_context, target_size)
        proj_body    = F.adaptive_avg_pool2d(proj_body, target_size)
        proj_face    = F.adaptive_avg_pool2d(proj_face, target_size)
        
        # Flatten spatial dimensions into tokens.
        N = target_size[0] * target_size[1]  # e.g., 49 tokens.
        tokens_context = proj_context.view(B, self.embed_dim, N).permute(0, 2, 1)
        tokens_body    = proj_body.view(B, self.embed_dim, N).permute(0, 2, 1)
        tokens_face    = proj_face.view(B, self.embed_dim, N).permute(0, 2, 1)
        
        # Concatenate face and body tokens along the sequence dimension.
        tokens_fb = torch.cat([tokens_body, tokens_face], dim=1)
        
        # Apply multihead attention: context as query and concatenated body+face as key/value.
        attn_output, _ = self.mha(query=tokens_context, key=tokens_fb, value=tokens_fb)
        tokens_fused = self.ln(tokens_context + attn_output)
        tokens_fused = tokens_fused + self.ffn(tokens_fused)
        
        # Average pool over tokens.
        fused_feature = tokens_fused.mean(dim=1)
        cat_out = self.fc_cat(fused_feature)
        cont_out = self.fc_cont(fused_feature)
        return cat_out if not isVADPrediction else cont_out

class FusionSimple(nn.Module):
    """
    Simple fusion by concatenating features from context, body, and face,
    then applying fully connected layers.
    """
    def __init__(self, num_context_features, num_body_features, num_face_features, num_classes=26, num_cont=3):
        super(FusionSimple, self).__init__()
        self.fc1 = nn.Linear(num_context_features + num_body_features + num_face_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc_cat = nn.Linear(256, num_classes)
        self.fc_cont = nn.Linear(256, num_cont)
        self.relu = nn.ReLU()
    
    def forward(self, x_context, x_body, x_face):
        context = x_context.view(x_context.size(0), -1)
        body = x_body.view(x_body.size(0), -1)
        face = x_face.view(x_face.size(0), -1)
        features = torch.cat((context, body, face), dim=1)
        out = self.fc1(features)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        cat_out = self.fc_cat(out)
        cont_out = self.fc_cont(out)
        return cat_out if not isVADPrediction else cont_out

class FusionAttention(nn.Module):
    """
    Fusion model using an attention mechanism over concatenated features.
    """
    def __init__(self, num_context_features, num_body_features, num_face_features, num_classes=26, num_cont=3):
        super(FusionAttention, self).__init__()
        total_features = num_context_features + num_body_features + num_face_features
        self.attention = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.ReLU(),
            nn.Linear(128, total_features),
            nn.Softmax(dim=1)
        )
        self.fc1 = nn.Linear(total_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc_cat = nn.Linear(256, num_classes)
        self.fc_cont = nn.Linear(256, num_cont)
        self.relu = nn.ReLU()
    
    def forward(self, x_context, x_body, x_face):
        context = x_context.view(x_context.size(0), -1)
        body = x_body.view(x_body.size(0), -1)
        face = x_face.view(x_face.size(0), -1)
        features = torch.cat((context, body, face), dim=1)
        attn_weights = self.attention(features)
        weighted = features * attn_weights
        out = self.fc1(weighted)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        cat_out = self.fc_cat(out)
        cont_out = self.fc_cont(out)
        return cat_out if not isVADPrediction else cont_out

class FusionTransformation(nn.Module):
    """
    Fusion model that applies individual transformations to each stream
    before concatenating them.
    """
    def __init__(self, num_context_features, num_body_features, num_face_features, num_classes=26, num_cont=3):
        super(FusionTransformation, self).__init__()
        self.trans_context = nn.Sequential(
            nn.Linear(num_context_features, 128),
            nn.ReLU()
        )
        self.trans_body = nn.Sequential(
            nn.Linear(num_body_features, 128),
            nn.ReLU()
        )
        self.trans_face = nn.Sequential(
            nn.Linear(num_face_features, 128),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(128 * 3, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc_cat = nn.Linear(256, num_classes)
        self.fc_cont = nn.Linear(256, num_cont)
        self.relu = nn.ReLU()
    
    def forward(self, x_context, x_body, x_face):
        context = x_context.view(x_context.size(0), -1)
        body = x_body.view(x_body.size(0), -1)
        face = x_face.view(x_face.size(0), -1)
        t_context = self.trans_context(context)
        t_body = self.trans_body(body)
        t_face = self.trans_face(face)
        features = torch.cat((t_context, t_body, t_face), dim=1)
        out = self.fc1(features)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        cat_out = self.fc_cat(out)
        cont_out = self.fc_cont(out)
        return cat_out if not isVADPrediction else cont_out

class FusionWeighted(nn.Module):
    """
    Fusion with learnable weights for each modality's contribution.
    Each modality's features are independently projected then 
    combined with learnable weights.
    """
    def __init__(self, num_context_features, num_body_features, num_face_features, num_classes=26, num_cont=3):
        super(FusionWeighted, self).__init__()
        
        # Project each modality to the same dimension
        self.context_proj = nn.Linear(num_context_features, 256)
        self.body_proj = nn.Linear(num_body_features, 256)
        self.face_proj = nn.Linear(num_face_features, 256)
        
        # Learnable weights for each modality
        self.context_weight = nn.Parameter(torch.ones(1))
        self.body_weight = nn.Parameter(torch.ones(1))
        self.face_weight = nn.Parameter(torch.ones(1))
        
        self.bn = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc_cat = nn.Linear(256, num_classes)
        self.fc_cont = nn.Linear(256, num_cont)
        self.relu = nn.ReLU()
        
    def forward(self, x_context, x_body, x_face):
        context = x_context.view(x_context.size(0), -1)
        body = x_body.view(x_body.size(0), -1)
        face = x_face.view(x_face.size(0), -1)
        
        # Project each modality
        context_feat = self.relu(self.context_proj(context))
        body_feat = self.relu(self.body_proj(body))
        face_feat = self.relu(self.face_proj(face))
        
        # Weighted combination
        weights_sum = self.context_weight + self.body_weight + self.face_weight + 1e-8
        fused = (context_feat * self.context_weight + 
                 body_feat * self.body_weight + 
                 face_feat * self.face_weight) / weights_sum
        
        out = self.bn(fused)
        out = self.dropout(out)
        
        cat_out = self.fc_cat(out)
        cont_out = self.fc_cont(out)
        
        return cat_out if not isVADPrediction else cont_out

class FusionGatedResidual(nn.Module):
    """
    Fusion with gated residual connections to control information flow
    between modalities.
    """
    def __init__(self, num_context_features, num_body_features, num_face_features, num_classes=26, num_cont=3):
        super(FusionGatedResidual, self).__init__()
        
        # Initial projections
        self.context_proj = nn.Linear(num_context_features, 256)
        self.body_proj = nn.Linear(num_body_features, 256)
        self.face_proj = nn.Linear(num_face_features, 256)
        
        # Gates for residual connections
        self.context_gate = nn.Sequential(
            nn.Linear(256, 256),
            nn.Sigmoid()
        )
        self.body_gate = nn.Sequential(
            nn.Linear(256, 256),
            nn.Sigmoid()
        )
        self.face_gate = nn.Sequential(
            nn.Linear(256, 256),
            nn.Sigmoid()
        )
        
        # Integration layers
        self.integration = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5)
        )
        
        self.fc_cat = nn.Linear(256, num_classes)
        self.fc_cont = nn.Linear(256, num_cont)
        self.relu = nn.ReLU()
        
    def forward(self, x_context, x_body, x_face):
        context = x_context.view(x_context.size(0), -1)
        body = x_body.view(x_body.size(0), -1)
        face = x_face.view(x_face.size(0), -1)
        
        # Initial features
        context_feat = self.relu(self.context_proj(context))
        body_feat = self.relu(self.body_proj(body))
        face_feat = self.relu(self.face_proj(face))
        
        # Initial fusion (simple average)
        fused = (context_feat + body_feat + face_feat) / 3.0
        
        # Apply gated residual connections
        context_gate_val = self.context_gate(fused)
        body_gate_val = self.body_gate(fused)
        face_gate_val = self.face_gate(fused)
        
        # Residual connections with gates
        fused = fused + context_gate_val * context_feat
        fused = fused + body_gate_val * body_feat
        fused = fused + face_gate_val * face_feat
        
        # Final integration
        out = self.integration(fused)
        
        cat_out = self.fc_cat(out)
        cont_out = self.fc_cont(out)
        
        return cat_out if not isVADPrediction else cont_out

class FusionFeaturePyramid(nn.Module):
    """
    Fusion by creating a pyramid of features at different abstraction levels.
    """
    def __init__(self, num_context_features, num_body_features, num_face_features, num_classes=26, num_cont=3):
        super(FusionFeaturePyramid, self).__init__()
        
        # Level 1: Direct projections
        self.context_l1 = nn.Linear(num_context_features, 128)
        self.body_l1 = nn.Linear(num_body_features, 128)
        self.face_l1 = nn.Linear(num_face_features, 128)
        
        # Level 2: More abstract representations
        self.l2_proj = nn.Sequential(
            nn.Linear(128*3, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Level 3: Even more abstract
        self.l3_proj = nn.Sequential(
            nn.Linear(128*4, 128),
            nn.ReLU()
        )
        
        self.final = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5)
        )
        
        self.fc_cat = nn.Linear(256, num_classes)
        self.fc_cont = nn.Linear(256, num_cont)
        self.relu = nn.ReLU()
        
    def forward(self, x_context, x_body, x_face):
        context = x_context.view(x_context.size(0), -1)
        body = x_body.view(x_body.size(0), -1)
        face = x_face.view(x_face.size(0), -1)
        
        # Level 1 features
        context_l1 = self.relu(self.context_l1(context))
        body_l1 = self.relu(self.body_l1(body))
        face_l1 = self.relu(self.face_l1(face))
        
        # Concatenate level 1 features
        l1_concat = torch.cat([context_l1, body_l1, face_l1], dim=1)
        
        # Level 2 features
        l2 = self.l2_proj(l1_concat)
        
        # Concatenate level 1 and level 2 features
        l2_concat = torch.cat([context_l1, body_l1, face_l1, l2], dim=1)
        
        # Level 3 features (most abstract)
        l3 = self.l3_proj(l2_concat)
        
        # Final processing
        out = self.final(l3)
        
        cat_out = self.fc_cat(out)
        cont_out = self.fc_cont(out)
        
        return cat_out if not isVADPrediction else cont_out

class FusionLowRankBilinear(nn.Module):
    """
    Fusion using low-rank bilinear pooling for efficient 
    cross-modal interactions.
    """
    def __init__(self, num_context_features, num_body_features, num_face_features, 
                 rank=64, num_classes=26, num_cont=3):
        super(FusionLowRankBilinear, self).__init__()
        
        # Reduce feature dimensions
        self.context_proj = nn.Linear(num_context_features, 256)
        self.body_proj = nn.Linear(num_body_features, 256)
        self.face_proj = nn.Linear(num_face_features, 256)
        
        # Low-rank projections
        self.context_U = nn.Linear(256, rank)
        self.context_V = nn.Linear(256, rank)
        self.body_U = nn.Linear(256, rank)
        self.body_V = nn.Linear(256, rank)
        self.face_U = nn.Linear(256, rank)
        self.face_V = nn.Linear(256, rank)
        
        # Integration
        self.final = nn.Sequential(
            nn.Linear(rank*3, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5)
        )
        
        self.fc_cat = nn.Linear(256, num_classes)
        self.fc_cont = nn.Linear(256, num_cont)
        self.relu = nn.ReLU()
        
    def forward(self, x_context, x_body, x_face):
        context = x_context.view(x_context.size(0), -1)
        body = x_body.view(x_body.size(0), -1)
        face = x_face.view(x_face.size(0), -1)
        
        # Project features
        context_feat = self.relu(self.context_proj(context))
        body_feat = self.relu(self.body_proj(body))
        face_feat = self.relu(self.face_proj(face))
        
        # Low-rank bilinear pooling
        context_bi = self.context_U(context_feat) * self.context_V(context_feat)
        body_bi = self.body_U(body_feat) * self.body_V(body_feat)
        face_bi = self.face_U(face_feat) * self.face_V(face_feat)
        
        # Concatenate bilinear features
        fused = torch.cat([context_bi, body_bi, face_bi], dim=1)
        
        # Final integration
        out = self.final(fused)
        
        cat_out = self.fc_cat(out)
        cont_out = self.fc_cont(out)
        
        return cat_out if not isVADPrediction else cont_out

class FusionEnsemble(nn.Module):
    """
    Fusion by creating an ensemble of simpler fusion approaches.
    """
    def __init__(self, num_context_features, num_body_features, num_face_features, num_classes=26, num_cont=3):
        super(FusionEnsemble, self).__init__()
        
        # Create multiple simple fusion modules
        self.fusion1 = FusionSimple(num_context_features, num_body_features, num_face_features, num_classes, num_cont)
        self.fusion2 = FusionWeighted(num_context_features, num_body_features, num_face_features, num_classes, num_cont)
        
        # Weights for ensemble combination
        self.weights = nn.Parameter(torch.ones(2)/2)
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x_context, x_body, x_face):
        # Get outputs from each fusion method
        out1 = self.fusion1(x_context, x_body, x_face)
        out2 = self.fusion2(x_context, x_body, x_face)
        
        # Normalize weights
        norm_weights = self.softmax(self.weights)
        
        # Weighted combination
        ensemble_out = norm_weights[0] * out1 + norm_weights[1] * out2
        
        return ensemble_out

class FusionLayerNorm(nn.Module):
    """
    Fusion with layer normalization applied to each modality
    before concatenation. This helps balance the contribution
    of each modality by normalizing their feature distributions.
    """
    def __init__(self, num_context_features, num_body_features, num_face_features, num_classes=26, num_cont=3):
        super(FusionLayerNorm, self).__init__()
        
        # Layer normalization for each modality
        self.context_norm = nn.LayerNorm(num_context_features)
        self.body_norm = nn.LayerNorm(num_body_features)
        self.face_norm = nn.LayerNorm(num_face_features)
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_context_features + num_body_features + num_face_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc_cat = nn.Linear(256, num_classes)
        self.fc_cont = nn.Linear(256, num_cont)
        self.relu = nn.ReLU()
        
    def forward(self, x_context, x_body, x_face):
        context = x_context.view(x_context.size(0), -1)
        body = x_body.view(x_body.size(0), -1)
        face = x_face.view(x_face.size(0), -1)
        
        # Apply layer normalization
        context = self.context_norm(context)
        body = self.body_norm(body)
        face = self.face_norm(face)
        
        features = torch.cat((context, body, face), dim=1)
        out = self.fc1(features)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        cat_out = self.fc_cat(out)
        cont_out = self.fc_cont(out)
        
        return cat_out if not isVADPrediction else cont_out

class FusionSelectiveDropout(nn.Module):
    """
    Fusion with different dropout rates applied to each modality.
    This can help the model learn more robust features by preventing
    over-reliance on any single modality.
    """
    def __init__(self, num_context_features, num_body_features, num_face_features, 
                 num_classes=26, num_cont=3, dropout_rates=[0.3, 0.3, 0.3]):
        super(FusionSelectiveDropout, self).__init__()
        
        # Dropouts for each modality
        self.context_dropout = nn.Dropout(dropout_rates[0])
        self.body_dropout = nn.Dropout(dropout_rates[1])
        self.face_dropout = nn.Dropout(dropout_rates[2])
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_context_features + num_body_features + num_face_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc_cat = nn.Linear(256, num_classes)
        self.fc_cont = nn.Linear(256, num_cont)
        self.relu = nn.ReLU()
        
    def forward(self, x_context, x_body, x_face):
        context = x_context.view(x_context.size(0), -1)
        body = x_body.view(x_body.size(0), -1)
        face = x_face.view(x_face.size(0), -1)
        
        # Apply selective dropout
        context = self.context_dropout(context)
        body = self.body_dropout(body)
        face = self.face_dropout(face)
        
        features = torch.cat((context, body, face), dim=1)
        out = self.fc1(features)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        cat_out = self.fc_cat(out)
        cont_out = self.fc_cont(out)
        
        return cat_out if not isVADPrediction else cont_out

class FusionBottleneck(nn.Module):
    """
    Fusion with dimensionality reduction applied to each modality.
    This can help identify and extract the most important features
    from each modality before fusion.
    """
    def __init__(self, num_context_features, num_body_features, num_face_features, 
                 bottleneck_dim=128, num_classes=26, num_cont=3):
        super(FusionBottleneck, self).__init__()
        
        # Bottleneck projections for each modality
        self.context_bottleneck = nn.Linear(num_context_features, bottleneck_dim)
        self.body_bottleneck = nn.Linear(num_body_features, bottleneck_dim)
        self.face_bottleneck = nn.Linear(num_face_features, bottleneck_dim)
        
        # Fully connected layers
        self.fc1 = nn.Linear(bottleneck_dim * 3, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc_cat = nn.Linear(256, num_classes)
        self.fc_cont = nn.Linear(256, num_cont)
        self.relu = nn.ReLU()
        
    def forward(self, x_context, x_body, x_face):
        context = x_context.view(x_context.size(0), -1)
        body = x_body.view(x_body.size(0), -1)
        face = x_face.view(x_face.size(0), -1)
        
        # Apply bottleneck projections
        context = self.relu(self.context_bottleneck(context))
        body = self.relu(self.body_bottleneck(body))
        face = self.relu(self.face_bottleneck(face))
        
        features = torch.cat((context, body, face), dim=1)
        out = self.fc1(features)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        cat_out = self.fc_cat(out)
        cont_out = self.fc_cont(out)
        
        return cat_out if not isVADPrediction else cont_out

class FusionWeightedWithNoise(nn.Module):
    """
    Weighted fusion with noise injection during training.
    Builds on the FusionWeighted model that showed improvement,
    but adds noise during training to improve robustness.
    """
    def __init__(self, num_context_features, num_body_features, num_face_features, 
                 num_classes=26, num_cont=3, noise_level=0.05):
        super(FusionWeightedWithNoise, self).__init__()
        
        # Learnable weights
        self.context_weight = nn.Parameter(torch.ones(1))
        self.body_weight = nn.Parameter(torch.ones(1))
        self.face_weight = nn.Parameter(torch.ones(1))
        
        self.noise_level = noise_level
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_context_features + num_body_features + num_face_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc_cat = nn.Linear(256, num_classes)
        self.fc_cont = nn.Linear(256, num_cont)
        self.relu = nn.ReLU()
        
    def forward(self, x_context, x_body, x_face):
        context = x_context.view(x_context.size(0), -1)
        body = x_body.view(x_body.size(0), -1)
        face = x_face.view(x_face.size(0), -1)
        
        # Get normalized weights (sum to 1)
        weights_sum = self.context_weight + self.body_weight + self.face_weight + 1e-8
        context_w = self.context_weight / weights_sum
        body_w = self.body_weight / weights_sum
        face_w = self.face_weight / weights_sum
        
        # Add noise during training
        if self.training:
            noise_c = torch.randn_like(context_w) * self.noise_level
            noise_b = torch.randn_like(body_w) * self.noise_level
            noise_f = torch.randn_like(face_w) * self.noise_level
            context_w = context_w + noise_c
            body_w = body_w + noise_b
            face_w = face_w + noise_f
            
            # Re-normalize after adding noise
            weights_sum = context_w + body_w + face_w + 1e-8
            context_w = context_w / weights_sum
            body_w = body_w / weights_sum
            face_w = face_w / weights_sum
        
        # Apply weighted concatenation
        context = context * context_w
        body = body * body_w
        face = face * face_w
        
        features = torch.cat((context, body, face), dim=1)
        out = self.fc1(features)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        cat_out = self.fc_cat(out)
        cont_out = self.fc_cont(out)
        
        return cat_out if not isVADPrediction else cont_out

class FusionHierarchical(nn.Module):
    """
    Hierarchical fusion that combines modalities in stages.
    First combines body and face (appearance-related), then
    fuses this with context (scene-related).
    """
    def __init__(self, num_context_features, num_body_features, num_face_features, 
                 num_classes=26, num_cont=3):
        super(FusionHierarchical, self).__init__()
        
        # First fusion stage (body and face)
        self.body_face_fusion = nn.Sequential(
            nn.Linear(num_body_features + num_face_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        
        # Second fusion stage (add context)
        self.fc1 = nn.Linear(256 + num_context_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc_cat = nn.Linear(256, num_classes)
        self.fc_cont = nn.Linear(256, num_cont)
        self.relu = nn.ReLU()
        
    def forward(self, x_context, x_body, x_face):
        context = x_context.view(x_context.size(0), -1)
        body = x_body.view(x_body.size(0), -1)
        face = x_face.view(x_face.size(0), -1)
        
        # First fusion: body and face
        body_face = torch.cat((body, face), dim=1)
        body_face_fused = self.body_face_fusion(body_face)
        
        # Second fusion: add context
        features = torch.cat((body_face_fused, context), dim=1)
        out = self.fc1(features)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        cat_out = self.fc_cat(out)
        cont_out = self.fc_cont(out)
        
        return cat_out if not isVADPrediction else cont_out

# Dictionary Mapping Fusion Method Names to Classes
fusion_dict = {
    'simple': FusionSimple,
    'attention': FusionAttention,
    'transformation': FusionTransformation,
    'cross_modal_transformer': FusionCrossModalTransformer,
    'weighted': FusionWeighted,
    'gated_residual': FusionGatedResidual,
    'feature_pyramid': FusionFeaturePyramid,
    'bilinear': FusionLowRankBilinear,
    'ensemble': FusionEnsemble,
    'layer_norm': FusionLayerNorm,
    'selective_dropout': FusionSelectiveDropout,
    'bottleneck': FusionBottleneck,
    'weighted_noise': FusionWeightedWithNoise,
    'hierarchical': FusionHierarchical,
}
