# Set up conda
source ~/miniconda3/etc/profile.d/conda.sh

# Create a conda environment
conda create -n habitat python=3.7 cmake=3.14.0 -y
conda activate habitat

# Setup habitat-sim
git clone --depth 1 --branch v0.2.2 https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim || exit
pip install -r requirements.txt
python setup.py install --headless
cd ..

# Install PyTorch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# Install additional Python packages
pip install gym==0.22.0 urllib3==1.25.11

# Install habitat-lab
git clone https://github.com/carlosgual/habitat-lab.git
cd habitat-lab || exit
python setup.py develop --all
cd ..

# Install wandb
pip install wandb

# Set environment variables to silence habitat-sim logs
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"
export HABITAT_SIM_LOG="quiet"
