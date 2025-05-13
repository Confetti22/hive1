python train_autoencoder.py  \
                        -gpus 0 \
                        -cfg 'config/vsi_ae_2d.yaml' \
                        -slurm \
                        -slurm_ngpus 1 \
                        -slurm_nnodes 1 \
                        -slurm_nodelist c003 \
                        -slurm_partition compute \