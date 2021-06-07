#!/bin/bash
#SBATCH -N 1
#SBATCH -c 14
#SBATCH --gres=gpu:4
#SBATCH -p long
#SBATCH --reservation=CHPS

#SBATCH --time=00:10:00

export PATH=$PATH:~/localinstall/bin:~/localinstall/usr/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/localinstall/usr/lib:~/localinstall/usr/lib64

make write_images=1
make run write_images=1