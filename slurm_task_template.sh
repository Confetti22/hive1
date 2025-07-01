#!/bin/bash

SCRIPT_PATH="/share/home/shiqiz/workspace/hive1/contrastive_train_on_whole_brain_cos_loss_3d.py"
SCRIPT_NAME=$(basename "$SCRIPT_PATH" .py)

sbatch --job-name="$SCRIPT_NAME" <<EOF
#!/bin/bash
#SBATCH --output=${SCRIPT_NAME}_%j.out
#SBATCH --error=${SCRIPT_NAME}_%j.err
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --partition=tao
#SBATCH --nodelist=t001
#SBATCH --gres=gpu:1

# Load environment
# eval "\$(conda shell.bash hook)"
# conda activate /share/home/shiqiz/.conda/envs/pytorch

# Run the script
python "$SCRIPT_PATH"
EOF