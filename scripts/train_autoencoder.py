import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import  DataLoader
from torch.utils.data.distributed import DistributedSampler

import argparse
import sys
import os
# Get the path to the parent directory of 'test', which is 'project'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

from lib.utils.distributed import init_dist_node, init_dist_gpu, get_shared_folder
from confettii.entropy_helper import entropy_filter


import submitit, random, sys
from pathlib import Path
import yaml
import shutil
import os

def auto_cast_config(config):
    for k, v in config.items():
        if isinstance(v, str):
            try:
                config[k] = float(v)
            except ValueError:
                pass
    return config


def parse_args():

    parser = argparse.ArgumentParser(description='Train Autoencoder (pipeline-config aware)')

    # === GENERAL === #
    parser.add_argument('-reset', action='store_true',
                                            help='Reset saved model logs and weights')
    parser.add_argument('-tb', action='store_true',
                                            help='Start TensorBoard')
    parser.add_argument('-gpus', type=str, default="0",
                                            help='GPUs list, only works if not on slurm')
    parser.add_argument('-cfg', type=str, help='Path to YAML config (pipeline.yaml or legacy flat YAML)',
                        default='config/pipeline.yaml')

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
        # Prefer loading via pipeline.py to get derived paths and consistency
        try:
            from pipeline import load_cfg as pipeline_load_cfg
            cfg = pipeline_load_cfg(args.cfg)
        except Exception:
            cfg = None

        def as_float_if_str(v):
            if isinstance(v, str):
                try:
                    return float(v)
                except ValueError:
                    return v
            return v

        if cfg and 'autoencoder' in cfg and 'paths' in cfg and 'run_id' in cfg:
            ae = cfg['autoencoder']
            R = cfg['_run']

            # Core training hyper-parameters mapped from pipeline.yaml
            mapped = {
                # I/O (use derived paths from pipeline.load_cfg)
                'out': str(R['ae_out_dir']),
                'exp_name': R['run_id'],
                # dataset paths for SimpleDataset
                'data_path_dir': str(R['ae_train_dir']),
                'valid_data_path_dir': str(R['ae_test_dir']),
                # dataset + model selections
                'model_name': ae['model_name'],
                'dataset_name': ae['dataset_name'],
                'architecture': ae['architecture'],
                'loss_name': ae['loss_name'],
                'trainer_name': ae['trainer_name'],
                # model hyperparams
                'in_channel': int(ae['in_channel']),
                'out_channel': int(ae['out_channel']),
                'filters': ae['filters'],
                'kernel_size': ae['kernel_size'],
                'pad_mode': ae['pad_mode'],
                'act_mode': ae['act_mode'],
                'norm_mode': ae['norm_mode'],
                'block_type': ae['block_type'],
                'downsample_strategy': ae['downsample_strategy'],
                # Optional decoder last layer activation control for AE_1 variants
                'last_layer_act': ae['last_layer_act'],
                'return_bottle_neck': ae['return_bottle_neck'],
                # training knobs
                'epoch_num': int(ae['epoch_num']),
                'save_every': int(ae['save_every']),
                'batch_per_gpu': int(ae['batch_size']),
                'fp16': bool(ae['fp16']),
                'start_epoch': int(ae['start_epoch']),
                # solver
                'optimizer': ae['optimizer'],
                'lr_scheduler': ae['lr_scheduler'],
                'weight_decay': as_float_if_str(ae['weight_decay']),
                'lr_start': as_float_if_str(ae['lr_start']),
                'lr_end': as_float_if_str(ae['lr_end']),
                'warmup_epochs': int(ae['warmup_epochs']),
                # misc defaults used by training code
                'num_workers': int(ae['num_workers']),
                'shuffle': True,
            }

            # Also expose top-level dims for downstream components if needed
            if 'dims' in cfg:
                mapped['dims'] = int(cfg['dims'])

            # Update args namespace
            args.__dict__.update(mapped)
        else:
            # Legacy flat YAML fallback: copy values directly when not provided on CLI
            with open(args.cfg, 'r') as f:
                yml = yaml.safe_load(f)

            for k in ['lr_start', 'lr_end', 'weight_decay']:
                if k in yml:
                    yml[k] = as_float_if_str(yml[k])

            cmd = [c[1:] for c in sys.argv if c[0] == '-']
            for k, v in yml.items():
                if k not in cmd:
                    args.__dict__[k] = v

    return args


class SLURM_Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):

        init_dist_node(self.args)
        train(None, self.args)


def main():

    args = parse_args()
    args.port = random.randint(49152,65535)
    # Persist the effective config for reproducibility
    cfg_save_path = f"{args.out}/logs"
    os.makedirs(cfg_save_path, exist_ok=True)
    try:
        shutil.copy2(args.cfg, f"{cfg_save_path}/cfg.yaml")
    except Exception:
        pass
    print("config has been saved")

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
        mp.spawn(train, args=(args,), nprocs=args.ngpus_per_node)

        # After training completes locally, materialize a pipeline-friendly checkpoint
        # as <ae_out_dir>/best.ckpt so the orchestrator can detect completion.
        try:
            weights_dir = Path(args.out) / 'weights' 
            latest = sorted(weights_dir.glob('Epoch_*.pth'))
            if latest:
                best_ckpt = Path(args.out) / 'best.ckpt'
                shutil.copy2(latest[-1], best_ckpt)
                print(f"Wrote best.ckpt → {best_ckpt}")
        except Exception as e:
            print(f"[WARN] best.ckpt generation failed: {e}")
	

def train(gpu, args):


    # === SET ENV === #
    init_dist_gpu(gpu, args)
    
    # === DATA === #
    # from lib.datasets.visor_3d_dataset import get_dataset
    get_dataset_fn = getattr(__import__("lib.datasets.{}".format(args.dataset_name), fromlist=["get_dataset"]), "get_dataset")
    #todo : verify the args.data_path_dir is correct
    print(f"Loading data from {args.data_path_dir} and {args.valid_data_path_dir}")
    train_dataset = get_dataset_fn(args.data_path_dir)
    train_sampler = DistributedSampler(train_dataset, shuffle=args.shuffle, num_replicas = args.world_size, rank = args.rank, seed = 31)
    train_loader = DataLoader(dataset=train_dataset, 
                            sampler = train_sampler,
                            batch_size=args.batch_per_gpu, 
                            num_workers= args.num_workers,
                            pin_memory = True,
                            drop_last = False, 
                            )
    valid_dataset  = get_dataset_fn(args.valid_data_path_dir)
    valid_sampler = DistributedSampler(valid_dataset, shuffle= args.shuffle, num_replicas = args.world_size, rank = args.rank, seed = 31)
    valid_loader = DataLoader(dataset=valid_dataset, 
                            sampler = valid_sampler,
                            batch_size=args.batch_per_gpu, 
                            num_workers= args.num_workers,
                            pin_memory = True,
                            drop_last = False
                            )
    
    print(f"Data loaded")

    # === MODEL === #
    from torchsummary import summary

    build_model_fn = getattr(__import__("lib.arch.{}".format(args.architecture), fromlist=["build_autoencoder_model"]), "build_autoencoder_model")
    model = build_model_fn(args)
    model=model.cuda(args.gpu)
    model.train()

    
    #print out model info
    print(model)
    summary(model,(1,64,64,64))

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
