git config --global user.name ob1
git config --global user.email ongjjbrandon@gmail.com

apt update
apt install ffmpeg libsm6 libxext6 tmux vim

mkdir -p /ob1_ws/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /ob1_ws/miniconda3/miniconda.sh
bash /ob1_ws/miniconda3/miniconda.sh -b -u -p /ob1_ws/miniconda3
rm /ob1_ws/miniconda3/miniconda.sh

source /ob1_ws/miniconda3/bin/activate

conda init --all

conda config --add envs_dirs /ob1_ws/conda/envs
conda config --add pkgs_dirs /ob1_ws/conda/pkgs

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
pip install --upgrade pip

pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install isaaclab[isaacsim,all]==2.3.0 --extra-index-url https://pypi.nvidia.com

# test isaacsim works
OMNI_KIT_ALLOW_ROOT=1 isaacsim

# kill isaacsim and install Gr00t
git config --global url."https://github.com/".insteadOf "git@github.com:"
git submodule update --init --recursive

# Within IsaacLabEvalTasks directory
cd submodules/Isaac-GR00T
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4
export PYTHONPATH=$PYTHONPATH:$INSTALL_DIR/IsaacLabEvalTasks/submodules/Isaac-GR00T

python -c "import gr00t; print('gr00t imported successfully')"
# Within IsaacLabEvalTasks directory
python -m pip install -e source/isaaclab_eval_tasks

curl -LsSf https://hf.co/cli/install.sh | bash
source ~/.bashrc
# export CKPT="nvidia/GR00T-N1-2B-tuned-Nut-Pouring-task"
export CKPT="nvidia/GR00T-N1-2B-tuned-Exhaust-Pipe-Sorting-task"
export CKPT_LOCAL_DIR="/ob1_ws/hf/ckpts/$CKPT"
hf download $CKPT --local-dir $CKPT_LOCAL_DIR

# Download dir to get the HDF5 files required to run sim eval
export DATASET="nvidia/PhysicalAI-GR00T-Tuned-Tasks"
export DATASET_ROOT_DIR="/ob1_ws/hf/datasets/PhysicalAI-GR00T-Tuned-Tasks/" 
hf download --repo-type dataset $DATASET --local-dir $DATASET_ROOT_DIR

pip install pin-pink 
conda install pinocchio -c conda-forge
conda install -c conda-forge libstdcxx-ng
pip install "numpy<2.0.0,>=1.23.5" "ml-dtypes~=0.2.0"
# Within IsaacLabEvalTasks directory
# Assume the post-trained policy checkpoint with model files are under CKPTS_PATH
export LD_LIBRARY_PATH=/ob1_ws/conda/envs/env_isaaclab/lib:$LD_LIBRARY_PATH
export CKPTS_PATH="/ob1_ws/hf/ckpts/GR00T-N1-2B-tuned-Nut-Pouring-task"
export EVAL_RESULTS_FNAME="/ob1_ws/IsaacLabEvalTasks/eval_results/eval_nutpouring.json"

python scripts/evaluate_gn1.py \
    --num_feedback_actions 16 \
    --num_envs 10 \
    --task_name nutpouring \
    --eval_file_path $EVAL_RESULTS_FNAME \
    --model_path $CKPTS_PATH \
    --rollout_length 30 \
    --seed 10 \
    --max_num_rollouts 100

export CKPTS_PATH="/ob1_ws/hf/ckpts/nvidia/GR00T-N1-2B-tuned-Exhaust-Pipe-Sorting-task"
export EVAL_RESULTS_FNAME="/ob1_ws/IsaacLabEvalTasks/eval_results/eval_pipesorting.json"
python scripts/evaluate_gn1.py \
    --num_feedback_actions 16 \
    --num_envs 10 \
    --task_name pipesorting \
    --eval_file_path $EVAL_RESULTS_FNAME \
    --checkpoint_name gr00t-n1-2b-tuned-pipesorting \
    --model_path $CKPTS_PATH \
    --rollout_length 20 \
    --seed 10 \
    --max_num_rollouts 100