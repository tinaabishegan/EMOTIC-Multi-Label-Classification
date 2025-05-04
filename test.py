import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from dataset import EmoticDataset, default_context_norm, default_body_norm, default_face_norm
from model import get_context_model, get_body_model, get_face_model
from fusion import fusion_dict
from utils import plot_confusion_matrix
import matplotlib.pyplot as plt
from augment import swint_none_transform, no_aug_transform

# Define identity transform.
identity = lambda x: x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_model", type=str, default="resnet50",
                        help="Context backbone model (e.g., 'resnet50' or 'swin')")
    parser.add_argument("--body_model", type=str, default="resnet101",
                        help="Body backbone model (e.g., 'resnet101' or 'swin')")
    parser.add_argument("--face_model", type=str, default="mobilenet_v3",
                        help="Face backbone model")
    parser.add_argument("--fusion", type=str, default="cross_modal_transformer",
                        choices=list(fusion_dict.keys()),
                        help="Fusion method")
    parser.add_argument("--results_dir", type=str, default="./models",
                        help="Directory where the saved model states are stored")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Automatically determine whether to use SwinT-specific transforms based on the backbone names.
    isContextSwint = args.context_model.lower() == "swin" or args.context_model.lower() == "deit"
    isBodySwint = args.body_model.lower() == "swin" or args.body_model.lower() == "deit"

    # Load test data.
    data_src = "./emotic"
    test_context = np.load(os.path.join(data_src, "emotic_pre", "test_context_arr.npy"))
    test_body    = np.load(os.path.join(data_src, "emotic_pre", "test_body_arr.npy"))
    test_cat     = np.load(os.path.join(data_src, "emotic_pre", "test_cat_arr.npy"))
    test_face    = np.stack((np.load(os.path.join(data_src, "emotic_pre", "test_face_arr.npy")),) * 3, axis=-1)
    test_cont    = np.load(os.path.join(data_src, "emotic_pre", "test_cont_arr.npy"))  # Even if not used

    # Set transformation pipelines.
    # If a backbone is Swin, use the swint_test_transform; otherwise, no transform is applied.
    transform_context = swint_none_transform if isContextSwint else no_aug_transform
    transform_body = swint_none_transform if isBodySwint else no_aug_transform
    transform_face = no_aug_transform  # Face images are already tensors.

    # Create test dataset.
    test_dataset = EmoticDataset(
        x_context=test_context,
        x_body=test_body,
        x_face=test_face,
        y_cat=test_cat,
        y_cont=test_cont,
        context_transform=transform_context,
        body_transform=transform_body,
        face_transform=transform_face,
        context_norm=default_context_norm,
        body_norm=default_body_norm,
        face_norm=default_face_norm,
        isContextSwint=isContextSwint,
        isBodySwint=isBodySwint
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load pretrained backbones.
    model_context, num_context_features = get_context_model(args.context_model)
    model_body, num_body_features = get_body_model(args.body_model)
    model_face, num_face_features = get_face_model(args.face_model)

    # Load fusion model.
    FusionClass = fusion_dict[args.fusion]
    fusion_model = FusionClass(num_context_features, num_body_features, num_face_features, num_classes=26)

    # Move models to device.
    model_context.to(device)
    model_body.to(device)
    model_face.to(device)
    fusion_model.to(device)

    # Load saved weights.
    model_context.load_state_dict(torch.load(os.path.join(args.results_dir, "model_context.pth"), map_location=device))
    model_body.load_state_dict(torch.load(os.path.join(args.results_dir, "model_body.pth"), map_location=device))
    model_face.load_state_dict(torch.load(os.path.join(args.results_dir, "model_face.pth"), map_location=device))
    fusion_model.load_state_dict(torch.load(os.path.join(args.results_dir, "fusion_model.pth"), map_location=device))

    # Set models to evaluation mode.
    model_context.eval()
    model_body.eval()
    model_face.eval()
    fusion_model.eval()

    # Evaluation loop.
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for (img_ctx, img_body, img_face, lbl_cat, _) in test_loader:
            img_ctx  = img_ctx.to(device)
            img_body = img_body.to(device)
            img_face = img_face.to(device)
            lbl_cat_np = lbl_cat.numpy()

            feat_ctx = model_context(img_ctx)
            feat_bdy = model_body(img_body)
            feat_fce = model_face(img_face)

            outputs = fusion_model(feat_ctx, feat_bdy, feat_fce).cpu().numpy()
            all_preds.append(outputs)
            all_labels.append(lbl_cat_np)

    all_preds = np.concatenate(all_preds, axis=0).T  # Shape: (num_classes, num_samples)
    all_labels = np.concatenate(all_labels, axis=0).T

    # Compute per-class Average Precision.
    from sklearn.metrics import average_precision_score, accuracy_score, f1_score, precision_score, recall_score
    per_class_ap = {}
    aps = []
    for i in range(26):
        ap = average_precision_score(all_labels[i, :], all_preds[i, :])
        per_class_ap[f"Emotion_{i}"] = ap
        aps.append(ap)
    mAP = np.mean(aps)
    print("Test mAP:", mAP)

    # For single-label metrics, compute predictions as argmax.
    pred_classes = np.argmax(all_preds, axis=0)
    true_classes = np.argmax(all_labels, axis=0)

    accuracy = accuracy_score(true_classes, pred_classes)
    precision = precision_score(true_classes, pred_classes, average="weighted", zero_division=0)
    recall = recall_score(true_classes, pred_classes, average="weighted", zero_division=0)
    f1 = f1_score(true_classes, pred_classes, average="weighted", zero_division=0)

    print("Test Accuracy:", accuracy)
    print("Test Precision:", precision)
    print("Test Recall:", recall)
    print("Test F1 Score:", f1)

    # Plot confusion matrix.
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_classes, pred_classes)
    plot_confusion_matrix(cm, classes=[f"Emotion {i}" for i in range(26)],
                          title="Confusion Matrix",
                          filename=os.path.join(args.results_dir, "confusion_matrix.png"))

    # Log metrics and per-emotion AP into a text file.
    metrics_file = os.path.join(args.results_dir, "evaluation_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("Evaluation Metrics:\n")
        f.write(f"Mean Average Precision (mAP): {mAP:.4f}\n")
        f.write("Per-Emotion AP:\n")
        for emo, ap in per_class_ap.items():
            f.write(f"  {emo}: {ap:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

if __name__ == "__main__":
    main()
