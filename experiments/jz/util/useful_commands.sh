module load anaconda-py3/2024.06

conda activate /lustre/fsn1/projects/rech/eqb/uye44va/conda_envs/feedback

# set virtual env path for Python scripts
export PATH=/lustre/fsn1/projects/rech/eqb/uye44va/conda_envs/feedback/bin:$PATH

idracct

srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=00:10:00 --account=eqb@a100 --pty bash

srun --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=00:10:00 --account=eqb@a100 --partition=gpu_p13 --pty bash

srun --pty --account=eqb@a100 --nodes=1 --ntasks-per-node=1 \
     --gres=gpu:1 --cpus-per-task=8 -C "a100" -t 00:10:00 bash -i


srun --pty --account=eqb@h100 --nodes=1 --ntasks-per-node=1 \
     --gres=gpu:1 --cpus-per-task=8 -C "h100" -t 01:00:00 bash -i