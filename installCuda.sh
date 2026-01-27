# For Ubuntu-24.04
sudo apt update && sudo apt upgrade -y
sudo apt install -y gcc g++ gfortran cmake-curses-gui build-essential bison flex m4
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-13-0  # For CUDA 13.0
rm cuda-keyring_1.1-1_all.deb
echo 'export PATH="/usr/local/cuda-13.0/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
