# runner_stage1.py
import os
import subprocess
import itertools

# Define candidate backbone names for stage 1 (adjust these lists as needed).
context_models = ['swin', 'efficientnet_b7','deit']
body_models = ['swin', 'efficientnet_b7', 'deit']
face_models = ['efficientnet_b7', 'mobilenet_v3']

# Fixed parameters for Stage 1.
fusion = "simple"
ablation = "all"
loss = "discrete"
optimizer = "adam"
adasyn_level = "none"
context_aug = "none"
body_aug = "none"
face_aug = "none"
epochs = 20

base_results_dir = "results_stage1"
os.makedirs(base_results_dir, exist_ok=True)

for ctx, body, face in itertools.product(context_models, body_models, face_models):
    if not(ctx == "swin" and body == "swin" and face == "efficientnet_b7"):
        continue
    exp_name = f"{ctx}_{body}_{face}_{fusion}_{ablation}_{loss}_{optimizer}_adasyn-{adasyn_level}_ctx-{context_aug}_body-{body_aug}_face-{face_aug}"
    exp_dir = os.path.join(base_results_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Train
    train_cmd = [
        "python", "train.py",
        "--context_model", ctx,
        "--body_model", body,
        "--face_model", face,
        "--fusion", fusion,
        "--ablation", ablation,
        "--loss", loss,
        "--optimizer", optimizer,
        "--epochs", str(epochs),
        "--adasyn_level", adasyn_level,
        "--context_aug_level", context_aug,
        "--body_aug_level", body_aug,
        "--face_aug_level", face_aug,
        "--model_dir", exp_dir
    ]
    print("Stage1 Training:", " ".join(train_cmd))
    subprocess.run(train_cmd, check=True)
    
    # Test
    test_cmd = [
        "python", "test.py",
        "--context_model", ctx,
        "--body_model", body,
        "--face_model", face,
        "--fusion", fusion,
        "--results_dir", exp_dir
    ]
    print("Stage1 Testing:", " ".join(test_cmd))
    subprocess.run(test_cmd, check=True)
