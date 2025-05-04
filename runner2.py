# runner_stage2.py
import os
import subprocess

# Best backbone combination from Stage 1 (update these as determined from your experiments).
best_ctx = "swin"
best_body = "swin"
best_face = "efficientnet_b7"

fusion_models = ["simple", "attention","transformation", "cross_modal_transformer","weighted","gated_residual","bilinear","layer_norm","bottleneck","hierarchical"]
ablation = "all"
loss = "discrete"
optimizer = "adam"
adasyn_level = "none"
ctx_aug = "none"
body_aug = "none"
face_aug = "none"
epochs = 20

base_results_dir = "results_stage2"
os.makedirs(base_results_dir, exist_ok=True)

for fusion in fusion_models:
    exp_name = f"{best_ctx}_{best_body}_{best_face}_{fusion}_{ablation}_{loss}_{optimizer}_adasyn-{adasyn_level}_ctx-{ctx_aug}_body-{body_aug}_face-{face_aug}"
    exp_dir = os.path.join(base_results_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Train
    train_cmd = [
        "python", "train.py",
        "--context_model", best_ctx,
        "--body_model", best_body,
        "--face_model", best_face,
        "--fusion", fusion,
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
    print("Stage2 Training:", " ".join(train_cmd))
    subprocess.run(train_cmd, check=True)
    
    # Test
    test_cmd = [
        "python", "test.py",
        "--context_model", best_ctx,
        "--body_model", best_body,
        "--face_model", best_face,
        "--fusion", fusion,
        "--results_dir", exp_dir
    ]
    print("Stage2 Testing:", " ".join(test_cmd))
    subprocess.run(test_cmd, check=True)
