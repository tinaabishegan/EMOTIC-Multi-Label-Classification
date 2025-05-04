# runner_stage3.py
import os
import subprocess

# Best combination from previous stages.
best_ctx = "swin"
best_body = "swin"
best_face = "efficientnet_b7"
best_fusion = "weighted"
ablation = "all"
optimizer = "adam"
adasyn_level = "none"
ctx_aug = "none"
body_aug = "none"
face_aug = "none"
epochs = 20

loss_functions = ["discrete", "focal", "bce", "asl", "mllsc","dice"]

base_results_dir = "results_stage3"
os.makedirs(base_results_dir, exist_ok=True)

for loss in loss_functions:
    exp_name = f"{best_ctx}_{best_body}_{best_face}_{best_fusion}_{ablation}_{loss}_{optimizer}_adasyn-{adasyn_level}_ctx-{ctx_aug}_body-{body_aug}_face-{face_aug}"
    exp_dir = os.path.join(base_results_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    train_cmd = [
        "python", "train.py",
        "--context_model", best_ctx,
        "--body_model", best_body,
        "--face_model", best_face,
        "--fusion", best_fusion,
        "--ablation", ablation,
        "--loss", loss,
        "--optimizer", optimizer,
        "--epochs", str(epochs),
        "--adasyn_level", adasyn_level,
        "--context_aug_level", ctx_aug,
        "--body_aug_level", body_aug,
        "--face_aug_level", face_aug,
        "--model_dir", exp_dir
    ]
    print("Stage3 Training:", " ".join(train_cmd))
    subprocess.run(train_cmd, check=True)
    
    test_cmd = [
        "python", "test.py",
        "--context_model", best_ctx,
        "--body_model", best_body,
        "--face_model", best_face,
        "--fusion", best_fusion,
        "--results_dir", exp_dir
    ]
    print("Stage3 Testing:", " ".join(test_cmd))
    subprocess.run(test_cmd, check=True)
