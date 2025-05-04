# runner_stage4.py
import os
import subprocess

# Best combination from previous stages (update accordingly).
best_ctx = "swin"
best_body = "swin"
best_face = "efficientnet_b7"
best_fusion = "weighted"
best_loss = "discrete"
ablation = "all"
optimizer = "adam"
adasyn_level = "none"
epochs = 20

# Only test three scenarios: all 'none', all 'basic', all 'full'
augmentation_levels = ["none","basic","full"]

base_results_dir = "results_stage4"
os.makedirs(base_results_dir, exist_ok=True)

for aug_level in augmentation_levels:
    ctx_aug = aug_level
    body_aug = aug_level
    face_aug = aug_level

    exp_name = f"{best_ctx}_{best_body}_{best_face}_{best_fusion}_{ablation}_{best_loss}_{optimizer}_adasyn-{adasyn_level}_ctx-{ctx_aug}_body-{body_aug}_face-{face_aug}"
    exp_dir = os.path.join(base_results_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Train
    train_cmd = [
        "python", "train.py",
        "--context_model", best_ctx,
        "--body_model", best_body,
        "--face_model", best_face,
        "--fusion", best_fusion,
        "--ablation", ablation,
        "--loss", best_loss,
        "--optimizer", optimizer,
        "--epochs", str(epochs),
        "--adasyn_level", adasyn_level,
        "--context_aug_level", ctx_aug,
        "--body_aug_level", body_aug,
        "--face_aug_level", face_aug,
        "--model_dir", exp_dir
    ]
    print("Stage 4 Training:", " ".join(train_cmd))
    subprocess.run(train_cmd, check=True)

    # Test
    test_cmd = [
        "python", "test.py",
        "--context_model", best_ctx,
        "--body_model", best_body,
        "--face_model", best_face,
        "--fusion", best_fusion,
        "--results_dir", exp_dir
    ]
    print("Stage 4 Testing:", " ".join(test_cmd))
    subprocess.run(test_cmd, check=True)
