python train_autoencoder.py  \
                        -gpus 0 \
                        -cfg 'config/rm009.yaml' \
                        -slurm \
                        -slurm_ngpus 1 \
                        -slurm_nnodes 1 \
                        -slurm_nodelist c002 \
                        -slurm_partition compute \