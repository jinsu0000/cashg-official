import argparse
import os
import torch
import torch.distributed as dist
from src.config.config_parser import load_config
from datetime import datetime

from src.utils.logger import print_once
from src.train.handwriting_generator_trainer import HandwritingGenerationTrainer


def setup_ddp():

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])


        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        print(f"[DDP] Initialized: rank={rank}/{world_size}, local_rank={local_rank}", flush=True)
        return True, rank, world_size, local_rank
    else:

        return False, 0, 1, 0


def cleanup_ddp(is_ddp):
    if is_ddp:
        dist.destroy_process_group()


def main():


    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print_once(f"[CUDA Config] PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (fragmentation 방지)")


    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print_once(f"[CUDA] Cache cleared before training")


    is_ddp, rank, world_size, local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    if rank == 0:
        print_once(f"Loaded config from: {args.config}")
        print_once(f"Training mode: {cfg.ENV.MODE}, Start Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if is_ddp:
            print_once(f"[DDP] Multi-GPU Training: {world_size} GPUs")
        else:
            print_once(f"[Single GPU] Training on cuda:0")

    trainer = HandwritingGenerationTrainer(cfg, is_ddp=is_ddp, rank=rank, world_size=world_size, local_rank=local_rank)

    if args.checkpoint is not None:
        step = trainer.load_checkpoint(args.checkpoint)
        if rank == 0:
            print_once(f"[main] Resumed from checkpoint {args.checkpoint} (step={step})")

    try:
        trainer.train()
    finally:
        cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()
