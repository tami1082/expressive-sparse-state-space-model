#!/bin/bash
#SBATCH --job-name=pdssm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G              #
#SBATCH --output=/data/horse/ws/yuti394h-repo/log/%x-%j.out  # 
#SBATCH --error=/data/horse/ws/yuti394h-repo/log/%x-%j.err   # 


source /home/yuti394h/activate_env.sh
source /home/yuti394h/env/lra/bin/activate

which python
python -c "import jax; print(jax.devices())"

# wandb agent yudou-tian/icl-lra/v1slyslzs
/home/h1/yuti394h/env/lra/bin/python /home/yuti394h/expressive-sparse-state-space-model/state_tracking_PyTorch/run_experiment.py --c parity_0

# {'pd', 'perm_only', 'diag_only', 'perm_static', 'pd_static'}