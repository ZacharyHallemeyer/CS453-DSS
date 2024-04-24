#!/bin/bash
#SBATCH --job-name=CS453_DSS

#SBATCH --output=/scratch/nauID/CS453_DSS.out
#SBATCH --error=/scratch/nauID/CS453_DSS.err

#SBATCH --time=01:00:00
#SBATCH --mem=8192
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C a100 #GPU Model: k80, p100, v100, a100

#SBATCH --account=cs453-spr24	

ARCH=80
MODE=0

module load cuda/11.7
nvcc -O3 -DMODE=$MODE -arch=compute_$ARCH -code=sm_$ARCH -lcuda -lineinfo -Xcompiler -fopenmp DSS.cu -o DSS

for TRIAL in 1 2 3
do
    echo "Trial = $TRIAL"
    srun ./DSS 100 2 10.0 xy100.txt
done
