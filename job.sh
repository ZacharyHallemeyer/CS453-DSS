#!/bin/bash
#SBATCH --job-name=CS453_DSS

#change to your NAU ID below
#SBATCH --output=/scratch/zmh47/CS453_DSS.out
#SBATCH --error=/scratch/zmh47/CS453_DSS.err


#SBATCH --time=00:06:00		#Job timelimit is 3 minutes
#SBATCH --mem=20000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C a100 #GPU Model: k80, p100, v100, a100

#SBATCH --account=cs453-spr24	

module load cuda/11.7
nvcc -O3 -arch=compute_80 -code=sm_80 -lcuda -lineinfo -Xcompiler -fopenmp DSS.cu -o DSS
#run your program

srun ./DSS 7490 135000 1000 10000.0 bee_dataset_1D_feature_vectors.txt
