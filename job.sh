#!/bin/bash
#SBATCH --job-name=CS453_DSS

#SBATCH --output=/scratch/nauID/CS453_DSS.out
#SBATCH --error=/scratch/nauID/CS453_DSS.err

#SBATCH --time=01:00:00
#SBATCH --mem=8192
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C v100 #GPU Model: k80, p100, v100, a100

#SBATCH --account=cs453-spr24

SRC=src
DATA=data
FILE=bee_dataset_1D_feature_vectors.txt

ARCH=70 # GPU ARCHs: a100: 80, v100: 70
MODE=0

N=100
DIM=2
E=10000.0


module load cuda/11.7

for MODE in 0 1
do
    nvcc -O3 -DMODE=$MODE -arch=compute_$ARCH -code=sm_$ARCH -lcuda -lineinfo -Xcompiler -fopenmp $SRC/kd_tree.cu $SRC/DSS.cu -o DSS

    for FILE in xy100.csv
    do
        for TRIAL in 1 2 3
        do
            echo -e "\n\nMode = $MODE, File = $FILE, N = $N, E = $E, Trial = $TRIAL"
            srun ./DSS $N $DIM $E $DATA/$FILE
        done
    done
done
