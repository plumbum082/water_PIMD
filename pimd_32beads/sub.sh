#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=eann_pimd_npt
#SBATCH -N 1 -c 5
#SBATCH --gres=gpu:4 
#SBATCH -o out -e err

module load cuda/11.4

export OPENMM_CPU_THREADS=1
export OMP_NUM_THREADS=1

scrdir=/tmp

# clean folder
rm $scrdir/ipi_unix_dmff
# rm $scrdir/ipi_unix_eann_*
echo "***** start time *****"
date

# cd  $SLURM_SUBMIT_DIR
# run server
bash run_server.sh &
sleep 30

# check socket
ls -l $scrdir

# run client
iclient=1
while [ $iclient -le 4 ];do
      # bash run_EANN.sh &
    export CUDA_VISIBLE_DEVICES=$((iclient-1))
    bash run_client_dmff.sh &
    iclient=$((iclient+1))
    sleep 1s
done
# bash run_client_dmff.sh & 

wait

echo "***** finish time *****"
date

sleep 1
 


