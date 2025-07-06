import json
import sys
import torch
import torch.optim as optim
from pathlib import Path
import argparse
from sconf import Config, dump_args
import utils
import numpy as np
from utils import Logger
from torchvision import transforms
from datasets import (load_lmdb, load_json, read_data_from_lmdb,
                      get_comb_trn_loader, get_cv_comb_loaders)
from trainer import load_checkpoint, CombinedTrainer
from model import generator_dispatch, disc_builder
from model.modules import weights_init
from evaluator import Evaluator

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set GPU device


def setup_args_and_config():
    """
    setup_args_and_configs
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("name")  # Experiment name
    parser.add_argument("config_paths", nargs="+", help="path/to/config.yaml")  
    parser.add_argument("--resume", default=None, help="path/to/saved/.pth")
    parser.add_argument("--use_unique_name", default=False, action="store_true",
                        help="whether to use name with timestamp")

    args, left_argv = parser.parse_known_args()
    assert not args.name.endswith(".yaml")

    # Load configuration files with defaults
    cfg = Config(*args.config_paths, default="cfgs/defaults.yaml",
                 colorize_modified_item=True)  
    cfg.argv_update(left_argv)

    # Setup working directories
    cfg.work_dir = Path(cfg.work_dir) 
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique experiment name
    if args.use_unique_name:  
        timestamp = utils.timestamp()
        unique_name = "{}_{}".format(timestamp, args.name)
    else:
        unique_name = args.name

    cfg.unique_name = unique_name
    cfg.name = args.name

    # Create required directories
    (cfg.work_dir / "logs").mkdir(parents=True, exist_ok=True) 
    (cfg.work_dir / "checkpoints" / unique_name).mkdir(parents=True, exist_ok=True)  

    # Validate configuration
    if cfg.save_freq % cfg.val_freq:  
        raise ValueError("save_freq has to be multiple of val_freq.")

    return args, cfg


def setup_transforms(cfg):
    """
    setup_transforms: Create image transformations for training and validation
    """
    size = cfg.input_size
    tensorize_transform = [transforms.Resize((size, size)), transforms.ToTensor()]  
    if cfg.dset_aug.normalize:  
        tensorize_transform.append(transforms.Normalize([0.5], [0.5]))
        cfg.g_args.dec.out = "tanh"

    trn_transform = transforms.Compose(tensorize_transform)  
    val_transform = transforms.Compose(tensorize_transform)  
    return trn_transform, val_transform   


def load_pretrain_vae_model(load_path='pretrained_weights/VQ-VAE_chn_.pth', gen=None): 
    print(load_path)
    vae_state_dict = torch.load(load_path, map_location=torch.device('cuda:0'))
    component_objects = vae_state_dict["_vq_vae._embedding.weight"]

    # Remove encoder weights that will be transferred to generator
    del_key = []
    for key, _ in vae_state_dict.items():
        if "encoder" in key:
            del_key.append(key)

    # Initialize generator content encoder with pretrained weights
    i = 0
    for param in gen.content_encoder.parameters():
        param.data = vae_state_dict[del_key[i]]
        i += 1
        param.requires_grad = False    

    return component_objects


def train(args, cfg, ddp_gpu=0):
    """
    train
    :param atgs:
    :param cfg:
    :param ddp_gpu:
    :return:
    """
    torch.cuda.set_device(ddp_gpu)
    # Setup logging
    logger_path = cfg.work_dir / "logs" / "{}.log".format(cfg.unique_name)
    logger = Logger.get(file_path=logger_path, level="info", colorize=True)  

    # Setup paths for outputs
    image_scale = 0.6
    writer_path = cfg.work_dir / "runs" / cfg.unique_name
    eval_image_path = cfg.work_dir / "images" / cfg.unique_name
    # writer = utils.TBDiskWriter(writer_path, eval_image_path, scale=image_scale)

    # Log run information
    args_str = dump_args(args) 
    logger.info("Run Argv:\n> {}".format(" ".join(sys.argv)))
    logger.info("Args:\n{}".format(args_str))
    logger.info("Configs:\n{}".format(cfg.dumps()))
    logger.info("Unique name: {}".format(cfg.unique_name))
    logger.info("Get dataset ...")

    content_font = cfg.content_font  # Font used for content representation

    # Setup data transformations
    trn_transform, val_transform = setup_transforms(cfg) 

    # Load dataset
    env = load_lmdb(cfg.data_path)  # LMDB environment
    env_get = lambda env, x, y, transform: transform(read_data_from_lmdb(env, f'{x}_{y}')['img'])

    data_meta = load_json(cfg.data_meta)  # load train.json

    # Select data loader functions
    get_trn_loader = get_comb_trn_loader
    get_cv_loaders = get_cv_comb_loaders
    Trainer = CombinedTrainer 

    # Create data loaders
    trn_dset, trn_loader = get_trn_loader(env,
                                          env_get,
                                          cfg,
                                          data_meta["train"],
                                          trn_transform,
                                          num_workers=cfg.n_workers,
                                          shuffle=True,
                                          drop_last=True)

    cv_loaders = get_cv_loaders(env,
                                env_get,
                                cfg,
                                data_meta,
                                val_transform,
                                num_workers=0,
                                shuffle=False,
                                drop_last=True)

    logger.info("Build Few-shot model ...") 
   
    # Initialize generator
    g_kwargs = cfg.get("g_args", {})
    g_cls = generator_dispatch()
    gen = g_cls(1, cfg.C, 1, cfg, **g_kwargs)
    gen.cuda()
    gen.apply(weights_init(cfg.init))  # Apply weight initialization

    # Load pretrained components
    logger.info("Load pre-train model...")
    component_objects = load_pretrain_vae_model(cfg.vae_pth, gen)

    # Initialize discriminator if GAN loss is used
    if cfg.gan_w > 0.:
        d_kwargs = cfg.get("d_args", {})
        disc = disc_builder(cfg.C, trn_dset.n_fonts, trn_dset.n_unis, **d_kwargs)
        disc.cuda()
        disc.apply(weights_init(cfg.init))
    else:
        disc = None

    # Setup optimizers and learning rate schedulers
    g_optim = optim.Adam(gen.parameters(), lr=cfg.g_lr, betas=cfg.adam_betas)
    d_optim = optim.Adam(disc.parameters(), lr=cfg.d_lr, betas=cfg.adam_betas)
    gen_scheduler = torch.optim.lr_scheduler.StepLR(g_optim, step_size=cfg['step_size'], gamma=cfg['gamma'])
    dis_scheduler = torch.optim.lr_scheduler.StepLR(d_optim, step_size=cfg['step_size'], gamma=cfg['gamma']) \
        if disc is not None else None

    # logger.info("Gen struct:{}"
    #             "Dis struct:{}"
    #             .format(gen, disc))

    # Handle resume from checkpoint
    st_step = 1
    if args.resume:
        st_step, loss = load_checkpoint(args.resume, gen, disc, g_optim, d_optim, gen_scheduler, dis_scheduler)
        logger.info("Resumed checkpoint from {} (Step {}, Loss {:7.3f})".format(
            args.resume, st_step - 1, loss))
        if cfg.overwrite:
            st_step = 1
        else:
            pass

    # Initialize evaluator   
    envaluator = Evaluator(env,
                           env_get,
                           cfg,
                           logger,
                           cfg.batch_size,
                           val_transform,
                           content_font,
                           use_half=cfg.use_half)

    # Create trainer
    trainer = Trainer(gen, disc, g_optim, d_optim, gen_scheduler, dis_scheduler,
                      logger, envaluator, cv_loaders, cfg)

    # Load character similarity data
    with open(cfg.sim_path, 'r+') as file:
        chars_sim = file.read()

    chars_sim_dict = json.loads(chars_sim) 

    # Start training
    trainer.train(trn_loader, st_step, cfg["iter"], component_objects, chars_sim_dict)


def main():
    """Main entry point"""
    args, cfg = setup_args_and_config()
    # Set random seeds for reproducibility
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    train(args, cfg)


if __name__ == "__main__":
    main()
