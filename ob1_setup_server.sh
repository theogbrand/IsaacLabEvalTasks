git config --global user.name ob1
git config --global user.email ongjjbrandon@gmail.com

apt update
apt install ffmpeg libsm6 libxext6 tmux

mkdir -p /workspace/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /workspace/miniconda3/miniconda.sh
bash /workspace/miniconda3/miniconda.sh -b -u -p /workspace/miniconda3
rm /workspace/miniconda3/miniconda.sh

source /workspace/miniconda3/bin/activate

conda init --all

conda config --add envs_dirs /workspace/conda/envs
conda config --add pkgs_dirs /workspace/conda/pkgs

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