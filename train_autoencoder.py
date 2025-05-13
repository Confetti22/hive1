import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import  DataLoader
from torch.utils.data.distributed import DistributedSampler

import argparse
from lib.utils.distributed import init_dist_node, init_dist_gpu, get_shared_folder
from confettii.entropy_helper import entropy_filter


import submitit, random, sys
from pathlib import Path


def parse_args():

    parser = argparse.ArgumentParser(description='Template')

    # === GENERAL === #
    parser.add_argument('-reset', action='store_true',
                                            help='Reset saved model logs and weights')
    parser.add_argument('-tb', action='store_true',
                                            help='Start TensorBoard')
    parser.add_argument('-gpus', type=str, default="0",
                                            help='GPUs list, only works if not on slurm')
    parser.add_argument('-cfg', type =str,help='Configuration file',
                        default='config/vsi_ae_2d.yaml')

    # === Trainer === #



    # === SLURM === #
    parser.add_argument('-slurm', action='store_true',
                                            help='Submit with slurm')
    parser.add_argument('-slurm_ngpus', type=int, default = 8,
                                            help='num of gpus per node')
    parser.add_argument('-slurm_nnodes', type=int, default = 2,
                                            help='number of nodes')
    parser.add_argument('-slurm_nodelist', default = None,
                                            help='slurm nodeslist. i.e. "GPU17,GPU18"')
    parser.add_argument('-slurm_partition', type=str, default = "general",
                                            help='slurm partition')
    parser.add_argument('-slurm_timeout', type=int, default = 2800,
                                            help='slurm timeout minimum, reduce if running on the "Quick" partition')


    args = parser.parse_args()

    # cmdline parameters will overwrite the CFG parameters

    # === Read CFG File === #
    if args.cfg:



        with open(args.cfg, 'r') as f:
            import yaml
            yml = yaml.safe_load(f)

        # update values from cfg file only if not passed in cmdline
        cmd = [c[1:] for c in sys.argv if c[0]=='-']
        for k,v in yml.items():
            if k not in cmd:
                args.__dict__[k] = v

    return args


class SLURM_Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):

        init_dist_node(self.args)
        train(None, self.args)


import shutil
import os
def main():

    args = parse_args()
    args.port = random.randint(49152,65535)


    cfg_save_path = f"{args.out}/logs/{args.exp_name}"
    os.makedirs(cfg_save_path,exist_ok=True)
    shutil.copy2(args.cfg,f"{cfg_save_path}/cfg.yaml")
    print(f"config has been saved")

    if args.slurm:

        # Almost copy-paste from https://github.com/facebookresearch/deit/blob/main/run_with_submitit.py
        args.output_dir = get_shared_folder(args) / "%j"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)

        executor.update_parameters(
            mem_gb=128*args.slurm_ngpus,
            gpus_per_node=args.slurm_ngpus,
            tasks_per_node=args.slurm_ngpus,
            cpus_per_task=2,
            nodes=args.slurm_nnodes,
            timeout_min=2800,
            slurm_partition=args.slurm_partition
        )

        if args.slurm_nodelist:
            executor.update_parameters(slurm_additional_parameters = {"nodelist": f'{args.slurm_nodelist}' })

        executor.update_parameters(name=args.exp_name)
        trainer = SLURM_Trainer(args)
        job = executor.submit(trainer)
        print(f"Submitted job_id: {job.job_id}")


    else:
        init_dist_node(args)
        mp.spawn(train, args=(args), nprocs=args.ngpus_per_node)
	

def train(gpu, args):


    # === SET ENV === #
    init_dist_gpu(gpu, args)
    
    # === DATA === #
    # from lib.datasets.visor_3d_dataset import get_dataset
    build_dataset_fn = getattr(__import__("lib.datasets.{}".format(args.dataset_name), fromlist=["get_dataset"]), "get_dataset")
    train_dataset = build_dataset_fn(args)

    train_sampler = DistributedSampler(train_dataset, shuffle=args.shuffle, num_replicas = args.world_size, rank = args.rank, seed = 31)
    train_loader = DataLoader(dataset=train_dataset, 
                            sampler = train_sampler,
                            batch_size=args.batch_per_gpu, 
                            num_workers= args.num_workers,
                            pin_memory = True,
                            drop_last = True
                            )
    get_valid_dataset = getattr(__import__("lib.datasets.{}".format(args.dataset_name), fromlist=["get_valid_dataset"]), "get_valid_dataset")
    valid_dataset  = get_valid_dataset(args)
    valid_sampler = DistributedSampler(valid_dataset, shuffle= args.shuffle, num_replicas = args.world_size, rank = args.rank, seed = 31)
    valid_loader = DataLoader(dataset=valid_dataset, 
                            sampler = valid_sampler,
                            batch_size= 6, 
                            num_workers= args.num_workers,
                            pin_memory = True,
                            drop_last = True
                            )
    
    print(f"Data loaded")

    # === MODEL === #
    from torchsummary import summary

    build_model_fn = getattr(__import__("lib.arch.{}".format(args.exp_name_name), fromlist=["build_autoencoder_model"]), "build_autoencoder_model")
    model = build_model_fn(args)
    model=model.cuda(args.gpu)
    model.train()

    
    #print out model info
    print(model)
    summary(model,(1,512,512))
    exit(0)

    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model) #group_norm did not require to sync, group_norm is preferred when batch_size is small
    model = nn.parallel.DistributedDataParallel(model, device_ids= [args.gpu],find_unused_parameters=True)

    # === LOSS === #
    get_loss = getattr(__import__("lib.loss.{}".format(args.loss_name), fromlist=["get_loss"]), "get_loss")
    loss = get_loss(args).cuda(args.gpu)
    
    #reconsturct the preprocess image in range [0-1] with bce loss

    # === OPTIMIZER === #
    from lib.core.optimizer import get_optimizer
    optimizer = get_optimizer(model, args)

    # === TRAINING === #
    Trainer = getattr(__import__("lib.trainers.{}".format(args.trainer_name), fromlist=["Trainer"]), "Trainer")
    Trainer(args, train_loader,valid_loader, model, loss, optimizer).fit()


if __name__ == "__main__":
    main()
