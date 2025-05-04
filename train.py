import os
import argparse
import numpy as np
import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import dataset and augmentation utilities.
from dataset import EmoticDataset, default_context_norm, default_body_norm, default_face_norm
from augment import (
    context_aug_levels,
    body_aug_levels,
    face_aug_levels,
    swint_context_aug_levels,
    swint_body_aug_levels,
    mini_batch_adasyn,
    full_adasyn,
    smote
)

# Import fusion model dictionary and pretrained backbone getters.
from fusion import fusion_dict
from model import get_context_model, get_body_model, get_face_model

# Import loss functions.
from loss import DiscreteLoss, MultiLabelFocalLoss, AsymmetricLoss, MLLSCLoss, WeightedFocalLoss, DiceLoss
import torch.nn.functional as F

# Import utility functions for plotting and evaluation.
from utils import plot_loss_curve, plot_metric_curve, test_scikit_ap

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion", type=str, default="cross_modal_transformer",
                        choices=list(fusion_dict.keys()),
                        help="Which fusion method to use")
    parser.add_argument("--context_model", type=str, default="resnet50",
                        help="Context backbone model (e.g. 'resnet50' or 'swin')")
    parser.add_argument("--body_model", type=str, default="resnet101",
                        help="Body backbone model (e.g. 'resnet101' or 'swin')")
    parser.add_argument("--face_model", type=str, default="mobilenet_v3",
                        help="Face backbone model")
    parser.add_argument("--loss", type=str, default="discrete",
                        choices=["discrete", "focal", "bce", "asl", "mllsc", "weighted_focal", "dice", "hybrid"],
                        help="Loss function for discrete classification")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "sgd"],
                        help="Optimizer")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--ablation", type=str, default="all",
                        choices=["all", "face", "body", "context", "face_body", "face_context", "body_context"],
                        help="Ablation setting")
    parser.add_argument("--model_dir", type=str, default="./models",
                        help="Directory where model checkpoints and plots will be saved")
    parser.add_argument("--adasyn_level", type=str, default="none",
                        choices=["none", "mini", "full", "smote"],
                        help="Dataset augmentation level: 'none', 'mini' (mini_batch_adasyn), 'full' (full_adasyn), or 'smote'")
    parser.add_argument("--context_aug_level", type=str, default="none",
                        choices=["none", "basic", "full"],
                        help="Image augmentation level for context stream")
    parser.add_argument("--body_aug_level", type=str, default="none",
                        choices=["none", "basic", "full"],
                        help="Image augmentation level for body stream")
    parser.add_argument("--face_aug_level", type=str, default="none",
                        choices=["none", "basic", "full"],
                        help="Image augmentation level for face stream")
    return parser.parse_args()

def main():
    args = parse_args()

    # Determine if the context or body backbones are Swin (case-insensitive).
    isContextSwint = args.context_model.lower() == "swin" or args.context_model.lower() == "deit"
    isBodySwint = args.body_model.lower() == "swin" or args.body_model.lower() == "deit"

    # Load Data Arrays
    data_src = "./emotic"
    train_context = np.load(os.path.join(data_src, "emotic_pre", "train_context_arr.npy"))
    train_body = np.load(os.path.join(data_src, "emotic_pre", "train_body_arr.npy"))
    train_cat = np.load(os.path.join(data_src, "emotic_pre", "train_cat_arr.npy"))
    train_cont = np.load(os.path.join(data_src, "emotic_pre", "train_cont_arr.npy"))
    val_context = np.load(os.path.join(data_src, "emotic_pre", "val_context_arr.npy"))
    val_body = np.load(os.path.join(data_src, "emotic_pre", "val_body_arr.npy"))
    val_cat = np.load(os.path.join(data_src, "emotic_pre", "val_cat_arr.npy"))
    val_cont = np.load(os.path.join(data_src, "emotic_pre", "val_cont_arr.npy"))
    
    # Face arrays: stack to create 3-channel images.
    train_face = np.stack((np.load(os.path.join(data_src, "emotic_pre", "train_face_arr.npy")),) * 3, axis=-1)
    val_face = np.stack((np.load(os.path.join(data_src, "emotic_pre", "val_face_arr.npy")),) * 3, axis=-1)

    print("Train shapes: context={}, body={}, face={}, cat={}".format(
        train_context.shape, train_body.shape, train_face.shape, train_cat.shape
    ))
    print("Val shapes: context={}, body={}, face={}, cat={}".format(
        val_context.shape, val_body.shape, val_face.shape, val_cat.shape
    ))

    # Setup Device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize Pretrained Backbones for Each Stream.
    model_context, num_context_features = get_context_model(args.context_model)
    model_body, num_body_features = get_body_model(args.body_model)
    model_face, num_face_features = get_face_model(args.face_model)

    # Freeze backbone parameters (we train only the fusion module).
    for param in model_context.parameters():
        param.requires_grad = False
    for param in model_body.parameters():
        param.requires_grad = False
    for param in model_face.parameters():
        param.requires_grad = False

    model_context.to(device)
    model_body.to(device)
    model_face.to(device)

    # Initialize Fusion Model.
    FusionClass = fusion_dict[args.fusion]
    fusion_model = FusionClass(num_context_features, num_body_features, num_face_features, num_classes=26)
    fusion_model.to(device)

    # Setup Loss Function.
    if args.loss == "discrete":
        loss_fn = DiscreteLoss(weight_type="dynamic", device=device)
    elif args.loss == "focal":
        loss_fn = MultiLabelFocalLoss()
    elif args.loss == "bce":
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == "asl":
        loss_fn = AsymmetricLoss(gamma_pos=0, gamma_neg=4)
    elif args.loss == "mllsc":
        loss_fn = MLLSCLoss()
    elif args.loss == "weighted_focal":
        class_counts = torch.sum(torch.tensor(train_cat), dim=0)
        weights = 1.0 / (class_counts + 1)
        weights = weights / weights.sum() * len(weights)
        loss_fn = WeightedFocalLoss(alpha=weights)
    elif args.loss == "dice":
        loss_fn = DiceLoss()

    # Setup Optimizer and Scheduler.
    if args.optimizer == "adam":
        optimizer = optim.Adam(fusion_model.parameters(), lr=1e-3, weight_decay=5e-4)
    else:
        optimizer = optim.SGD(fusion_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # Training Loop Variables.
    train_losses = []
    val_losses = []
    train_maps = []
    val_maps = []
    best_val_loss = float('inf')
    os.makedirs(args.model_dir, exist_ok=True)

    for epoch in range(args.epochs):
        # Set a different random seed for each epoch to ensure different augmentations
        epoch_seed = 42 + epoch
        torch.manual_seed(epoch_seed)
        np.random.seed(epoch_seed)
        
        # Re-create transformations with new random seed for each epoch
        if isContextSwint:
            transform_context = swint_context_aug_levels[args.context_aug_level]
        else:
            transform_context = context_aug_levels[args.context_aug_level]
        
        if isBodySwint:
            transform_body = swint_body_aug_levels[args.body_aug_level]
        else:
            transform_body = body_aug_levels[args.body_aug_level]
        
        transform_face = face_aug_levels[args.face_aug_level]
        
        # Create dataset objects with fresh augmentations for this epoch
        train_dataset = EmoticDataset(
            x_context=train_context,
            x_body=train_body,
            x_face=train_face,
            y_cat=train_cat,
            y_cont=train_cont,
            context_transform=transform_context,
            body_transform=transform_body,
            face_transform=transform_face,
            context_norm=default_context_norm,
            body_norm=default_body_norm,
            face_norm=default_face_norm,
            isContextSwint=isContextSwint,
            isBodySwint=isBodySwint
        )
        
        val_dataset = EmoticDataset(
            x_context=val_context,
            x_body=val_body,
            x_face=val_face,
            y_cat=val_cat,
            y_cont=val_cont,
            context_transform=transform_context,
            body_transform=transform_body,
            face_transform=transform_face,
            context_norm=default_context_norm,
            body_norm=default_body_norm,
            face_norm=default_face_norm,
            isContextSwint=isContextSwint,
            isBodySwint=isBodySwint
        )
        
        # Create new data loaders for this epoch
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Training loop
        fusion_model.train()
        running_loss = 0.0
        
        for i, (img_ctx, img_body, img_face, lbl_cat, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            img_ctx = img_ctx.to(device)
            img_body = img_body.to(device)
            img_face = img_face.to(device)
            lbl_cat = lbl_cat.to(device)
            
            optimizer.zero_grad()
            
            # Extract features from frozen backbones.
            with torch.no_grad():
                feat_context = model_context(img_ctx)
                feat_body = model_body(img_body)
                feat_face = model_face(img_face)
            
            # Apply ablation settings.
            if args.ablation == "all":
                pass
            elif args.ablation == "face":
                feat_context = torch.zeros_like(feat_context)
                feat_body = torch.zeros_like(feat_body)
            elif args.ablation == "body":
                feat_context = torch.zeros_like(feat_context)
                feat_face = torch.zeros_like(feat_face)
            elif args.ablation == "context":
                feat_body = torch.zeros_like(feat_body)
                feat_face = torch.zeros_like(feat_face)
            elif args.ablation == "face_body":
                feat_context = torch.zeros_like(feat_context)
            elif args.ablation == "face_context":
                feat_body = torch.zeros_like(feat_body)
            elif args.ablation == "body_context":
                feat_face = torch.zeros_like(feat_face)
                
            outputs = fusion_model(feat_context, feat_body, feat_face)
            
            # Apply dataset augmentation via ADASYN if selected.
            if args.adasyn_level != "none":
                X_features = torch.cat([
                    feat_context.view(feat_context.size(0), -1),
                    feat_body.view(feat_body.size(0), -1),
                    feat_face.view(feat_face.size(0), -1)
                ], dim=1)
                X_np = X_features.cpu().numpy()
                y_np = lbl_cat.cpu().numpy()
                
                if args.adasyn_level == "mini":
                    X_aug_np, y_aug_np = mini_batch_adasyn(X_np, y_np, k_neighbors=5)
                elif args.adasyn_level == "full":
                    X_aug_np, y_aug_np = full_adasyn(X_np, y_np, k_neighbors=5)
                elif args.adasyn_level == "smote":
                    X_aug_np, y_aug_np = smote(X_np, y_np, k_neighbors=5)
                    
                if X_aug_np.shape[0] > X_np.shape[0]:  # Only add if we actually generated synthetic samples
                    X_aug = torch.tensor(X_aug_np, dtype=torch.float32).to(device)
                    y_aug = torch.tensor(y_aug_np, dtype=torch.float32).to(device)
                    
                    d_ctx = feat_context.view(feat_context.size(0), -1).shape[1]
                    d_body = feat_body.view(feat_body.size(0), -1).shape[1]
                    d_face = feat_face.view(feat_face.size(0), -1).shape[1]
                    
                    # Ensure proper reshaping based on the original tensor shapes
                    X_ctx_aug = X_aug[:, :d_ctx].view(X_aug.size(0), *feat_context.shape[1:])
                    X_body_aug = X_aug[:, d_ctx:d_ctx+d_body].view(X_aug.size(0), *feat_body.shape[1:])
                    X_face_aug = X_aug[:, d_ctx+d_body:].view(X_aug.size(0), *feat_face.shape[1:])
                    
                    # Generate outputs for synthetic samples
                    outputs_syn = fusion_model(X_ctx_aug, X_body_aug, X_face_aug)
                    
                    # Combine original and synthetic outputs and labels
                    outputs = torch.cat([outputs, outputs_syn], dim=0)
                    lbl_cat = torch.cat([lbl_cat, y_aug], dim=0)
            
            loss = loss_fn(outputs, lbl_cat)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        scheduler.step()
        
        epoch_loss = running_loss
        train_losses.append(epoch_loss)

        # Evaluate on Training Set (compute mAP).
        fusion_model.eval()
        all_preds = []
        all_lbls = []
        with torch.no_grad():
            for (img_ctx, img_body, img_face, lbl_cat, _) in train_loader:
                img_ctx, img_body, img_face = img_ctx.to(device), img_body.to(device), img_face.to(device)
                lbl_cat_np = lbl_cat.numpy()

                feat_context = model_context(img_ctx)
                feat_body = model_body(img_body)
                feat_face = model_face(img_face)

                # Apply ablation settings.
                if args.ablation == "face":
                    feat_context = torch.zeros_like(feat_context)
                    feat_body = torch.zeros_like(feat_body)
                elif args.ablation == "body":
                    feat_context = torch.zeros_like(feat_context)
                    feat_face = torch.zeros_like(feat_face)
                elif args.ablation == "context":
                    feat_body = torch.zeros_like(feat_body)
                    feat_face = torch.zeros_like(feat_face)
                elif args.ablation == "face_body":
                    feat_context = torch.zeros_like(feat_context)
                elif args.ablation == "face_context":
                    feat_body = torch.zeros_like(feat_body)
                elif args.ablation == "body_context":
                    feat_face = torch.zeros_like(feat_face)

                preds = fusion_model(feat_context, feat_body, feat_face).cpu().numpy()
                all_preds.append(preds)
                all_lbls.append(lbl_cat_np)

        all_preds = np.concatenate(all_preds, axis=0).T
        all_lbls = np.concatenate(all_lbls, axis=0).T
        train_mAP = test_scikit_ap(all_preds, all_lbls)
        train_maps.append(train_mAP)

        # Validation Evaluation.
        val_running_loss = 0.0
        all_preds_val = []
        all_lbls_val = []
        with torch.no_grad():
            for (img_ctx, img_body, img_face, lbl_cat, _) in val_loader:
                img_ctx, img_body, img_face = img_ctx.to(device), img_body.to(device), img_face.to(device)
                lbl_cat = lbl_cat.to(device)

                feat_context = model_context(img_ctx)
                feat_body = model_body(img_body)
                feat_face = model_face(img_face)

                if args.ablation == "face":
                    feat_context = torch.zeros_like(feat_context)
                    feat_body = torch.zeros_like(feat_body)
                elif args.ablation == "body":
                    feat_context = torch.zeros_like(feat_context)
                    feat_face = torch.zeros_like(feat_face)
                elif args.ablation == "context":
                    feat_body = torch.zeros_like(feat_body)
                    feat_face = torch.zeros_like(feat_face)
                elif args.ablation == "face_body":
                    feat_context = torch.zeros_like(feat_context)
                elif args.ablation == "face_context":
                    feat_body = torch.zeros_like(feat_body)
                elif args.ablation == "body_context":
                    feat_face = torch.zeros_like(feat_face)

                outputs = fusion_model(feat_context, feat_body, feat_face)
                loss_val = loss_fn(outputs, lbl_cat)
                val_running_loss += loss_val.item()

                all_preds_val.append(outputs.cpu().numpy())
                all_lbls_val.append(lbl_cat.cpu().numpy())

        val_losses.append(val_running_loss)
        all_preds_val = np.concatenate(all_preds_val, axis=0).T
        all_lbls_val = np.concatenate(all_lbls_val, axis=0).T
        val_mAP = test_scikit_ap(all_preds_val, all_lbls_val)
        val_maps.append(val_mAP)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {epoch_loss:.3f}, Train mAP: {train_mAP:.3f}, "
              f"Val Loss: {val_running_loss:.3f}, Val mAP: {val_mAP:.3f}")

        # Save the model at the final epoch.
        if epoch + 1 == args.epochs:
            print("Model Saved")
            torch.save(model_context.state_dict(), os.path.join(args.model_dir, "model_context.pth"))
            torch.save(model_body.state_dict(), os.path.join(args.model_dir, "model_body.pth"))
            torch.save(model_face.state_dict(), os.path.join(args.model_dir, "model_face.pth"))
            torch.save(fusion_model.state_dict(), os.path.join(args.model_dir, "fusion_model.pth"))
        
    # Plot Loss and Metric Curves.
    plot_loss_curve(train_losses, val_losses, filename=os.path.join(args.model_dir, "loss_curve.png"))
    plot_metric_curve(train_maps, val_maps, filename=os.path.join(args.model_dir, "metric_curve.png"))

if __name__ == "__main__":
    main()
