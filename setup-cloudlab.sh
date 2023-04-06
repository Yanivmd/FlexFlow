#!/bin/bash

set -x

if [ $# -lt 2 ]; then
    echo "Usage: bash setup-cloudlab.sh username IP"
    exit
fi

ssh_arg="-o ConnectTimeout=10 -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no"
cudnn_file=cudnn-linux-x86_64-8.8.0.121_cuda11-archive.tar.xz
cudnn_dir=~/Downloads
work_dir=/mnt
ssh_down=255
ssh_up=0

username=$1
ip=$2

check_connectivity() {
    local status=$1
    local ip=$2
    while true
    do
        ssh $ssh_arg -q $username@$ip exit
        if [ $? -eq $status ]
        then
            break 1
        fi
    done
}

# Step 0: Configure Haoyu's personal shell
ssh $ssh_arg $username@$ip "git clone https://github.com/Lei-Houjyu/personal_shell.git; cd personal_shell; bash install.sh > /dev/null 2>&1"

# Step 1: Install CUDA 11.7
# Step 1.1: Disable Nouveau
ssh $ssh_arg $username@$ip "echo -e 'blacklist nouveau\noptions nouveau modeset=0' | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf"
ssh $ssh_arg $username@$ip "sudo update-initramfs -u"
ssh $ssh_arg $username@$ip "sudo reboot"

check_connectivity $ssh_down $ip
check_connectivity $ssh_up   $ip

# Step 1.2: Increase the disk capacity
ssh $ssh_arg $username@$ip "yes | sudo mkfs.ext4 /dev/sdb"
ssh $ssh_arg $username@$ip "sudo mount /dev/sdb $work_dir"
ssh $ssh_arg $username@$ip "sudo chown -R $username $work_dir"

# Step 1.3: Run CUDA runfile
ssh $ssh_arg $username@$ip "wget -P $work_dir https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run > /dev/null 2>&1"
ssh $ssh_arg $username@$ip "mkdir $work_dir/cuda $work_dir/tmp"
ssh $ssh_arg $username@$ip "sudo sh $work_dir/cuda_11.7.0_515.43.04_linux.run --silent --installpath=$work_dir/cuda --tmpdir=$work_dir/tmp"
ssh $ssh_arg $username@$ip "echo -e 'export PATH=/usr/local/cuda/bin:$PATH\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc"

# Step 2: Install cuDNN
ssh $ssh_arg $username@$ip "sudo apt update > /dev/null 2>&1; sudo apt install -y zlib1g > /dev/null 2>&1"
scp $ssh_arg $cudnn_dir/$cudnn_file $username@$ip:$work_dir
ssh $ssh_arg $username@$ip "cd $work_dir; tar -xvf $cudnn_file"
ssh $ssh_arg $username@$ip "sudo cp $work_dir/cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include"
ssh $ssh_arg $username@$ip "sudo cp -P $work_dir/cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64"
ssh $ssh_arg $username@$ip "sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*"

# Step 3: Install conda
ssh $ssh_arg $username@$ip "cd $work_dir; curl -O https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh > /dev/null 2>&1"
ssh $ssh_arg $username@$ip "bash $work_dir/Anaconda3-2022.10-Linux-x86_64.sh -b"

# Step 4: Build FlexFlow
ssh $ssh_arg $username@$ip "git clone --recursive -b osdi2022ae https://github.com/flexflow/FlexFlow.git $work_dir/FlexFlow"
ssh $ssh_arg $username@$ip "cd $work_dir/FlexFlow; sudo apt install -y pip libhdf5-dev > /dev/null 2>&1; pip install -r requirements.txt; mkdir build; ../config/config.linux; make -j 40"