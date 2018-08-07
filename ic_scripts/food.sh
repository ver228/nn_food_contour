#!/bin/sh
#PBS -l walltime=48:00:00
## This tells the batch manager to limit the walltime for the job to XX hours, YY minutes and ZZ seconds.

#PBS -l select=1:ncpus=2:mem=8gb:ngpus=1
## This tells the batch manager to use NN node with MM cpus and PP gb of memory per node with QQ gpus available.

#PBS -q gpgpu
## This tells the batch manager to enqueue the job in the general gpgpu queue.

module load anaconda3
module load cuda
source activate train-env

SRC=$WORK/food/train_set/
DST=$TMPDIR/train_set

rsync -ahW --no-compress $SRC $DST
du -hs $DST

python -c "import torch; print(torch.cuda.is_available())"

