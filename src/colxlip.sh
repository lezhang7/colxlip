#!/bin/bash
#SBATCH --job-name=colxlip-v1
#SBATCH --partition=long                         # Ask for unkillable job
#SBATCH --cpus-per-task=24                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=128G           
#SBATCH --time=12:00:00                                    
#SBATCH --output=./slurm_logs/colxlip/job_output-%j.txt
#SBATCH --error=./slurm_logs/colxlip/job_error-%j.txt 

module load miniconda/3
conda init
conda activate openflamingo

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo master port is $MASTER_PORT
export WORLD_SIZE=$SLURM_NTASKS_PER_NODE
echo world size is $WORLD_SIZE
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo master addr is $MASTER_ADDR
export OMP_NUM_THREADS=12

echo gpu_num is $SLURM_GPUS_ON_NODE

export NCCL_DEBUG=INFO

# Copy dataset to local storage for faster access
echo "Copying dataset to $SLURM_TMPDIR..."
mkdir -p $SLURM_TMPDIR/datasets/scraped_cc3m
cp $SCRATCH/openclip/datasets/scraped_cc3m/{00000..00574}.tar $SLURM_TMPDIR/datasets/scraped_cc3m/
echo "Dataset copy complete!"

torchrun --master_port $MASTER_PORT  --nproc_per_node=$SLURM_GPUS_ON_NODE -m main \
    --logs-dir ./logs \
    --model ViT-B-32-colxlip \
    --pretrained laion400m_e32 \
    --train-dataset-type webdataset  \
    --report-to wandb \
    --wandb-project-name colxlip \
    --name colxlip-v2-non-linear \
    --lr 1e-05 \
    --warmup 2000 \
    --epochs 32  \
    --caption-sampling-mode diverse_sampling \
    --num-sampled-captions 8 \
    --log-every-n-steps 2 \
    --train-data "$SLURM_TMPDIR/datasets/scraped_cc3m/{00000..00574}.tar"  \
    --train-num-samples 2823019 \
    --batch-size 512 \
    --precision amp \
    --workers 8 \
    --beta1 0.9 \
    --beta2 0.98 \
    --wd 0.1 \
    --eps 1e-6 \
    --alpha 0.7
