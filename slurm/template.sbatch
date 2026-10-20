#!/bin/bash
#SBATCH --job-name=gpu-vector-search
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err

# TODO: add Euler partition, GPU, time, module, and experiment-specific settings.