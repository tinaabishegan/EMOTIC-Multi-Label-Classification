import os
import subprocess
import sys # To check python executable

# Best combination from previous stages (as defined by user)
best_ctx = "swin"
best_body = "swin"
best_face = "efficientnet_b7"
best_fusion = "weighted"
best_loss = "discrete"
# Best image augmentation levels (used for 'none'/'mini' runs)
best_ctx_aug = "basic"
best_body_aug = "basic"
best_face_aug = "basic"
extraction_aug_level = "basic"

ablation = "all"
optimizer = "adam"
epochs = 20

# Dataset augmentation levels to test
dataset_aug_levels = ["mini"]

base_results_dir = "results_stage5" # Use a new base dir
os.makedirs(base_results_dir, exist_ok=True)

# Use the same python executable that runs this script
python_executable = sys.executable

for adasyn_level in dataset_aug_levels:
    print(f"\n--- Running Stage 5 for adasyn_level: {adasyn_level} ---")

    # --- Determine script and arguments based on adasyn_level ---
    common_args = [
        "--context_model", best_ctx,
        "--body_model", best_body,
        "--face_model", best_face,
        "--fusion", best_fusion,
        "--ablation", ablation,
        "--loss", best_loss,
        "--optimizer", optimizer,
        "--epochs", str(epochs),
    ]

    script_to_run = "train.py"
    # Experiment name reflects 'none' adasyn and specific image aug levels used
    exp_name = f"{best_ctx}_{best_body}_{best_face}_{best_fusion}_{ablation}_{best_loss}_{optimizer}_adasyn-{adasyn_level}_ctx-{best_ctx_aug}_body-{best_body_aug}_face-{best_face_aug}"
    exp_dir = os.path.join(base_results_dir, exp_name)
    
    train_args = common_args + [
        "--adasyn_level", adasyn_level,
        "--context_aug_level", best_ctx_aug, # Pass image aug levels
        "--body_aug_level", best_body_aug,
        "--face_aug_level", best_face_aug,
        "--model_dir", exp_dir
    ]

    os.makedirs(exp_dir, exist_ok=True)

    # --- Construct and Run Training Command ---
    train_cmd = [python_executable, script_to_run] + train_args
    print("\nStage5 Training Command:", " ".join(train_cmd))
    try:
        subprocess.run(train_cmd, check=True, text=True) # Added text=True for better output if needed
    except subprocess.CalledProcessError as e:
        print(f"ERROR during training for {exp_name}: {e}")
        continue # Skip testing if training failed

    # --- Construct and Run Testing Command ---
    test_cmd = [
        python_executable, "test.py",
        "--context_model", best_ctx,
        "--body_model", best_body,
        "--face_model", best_face,
        "--fusion", best_fusion,
        # Assuming test.py loads model from 'fusion_model_best*.pth' or 'fusion_model_final*.pth'
        "--results_dir", exp_dir
    ]
    print("\nStage5 Testing Command:", " ".join(test_cmd))
    try:
        subprocess.run(test_cmd, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR during testing for {exp_name}: {e}")

print("\n--- Stage 5 Runner Finished ---")
