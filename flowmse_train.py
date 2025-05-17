import argparse
from argparse import ArgumentParser
import os
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from flowmse.backbones.shared import BackboneRegistry
from flowmse.data_module import SpecsDataModule
from flowmse.odes import ODERegistry
from flowmse.model import VFModel

from datetime import datetime
import pytz

# 设置中国时区（北京时间）
china_tz = pytz.timezone('Asia/Shanghai')
current_time = datetime.now(china_tz)
timestamp = current_time.strftime("%Y%m%d%H%M%S")

# GPU设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def get_argparse_groups(parser):
    """解析参数分组"""
    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)
    return groups

def configure_tensorboard_logger(args, ode_class, dataset_name):
    """配置TensorBoard日志记录器"""
    log_dir_name = f"{dataset_name}_{timestamp}_{ode_class.__name__}"
    
    # 根据不同ODE类型添加特定参数到日志目录名
    if ode_class.__name__ == "FLOWMATCHING":
        log_dir_name += f"_sigma_{args.sigma_min}-{args.sigma_max}_Trev_{args.T_rev}"
    elif ode_class.__name__ == "STOCHASTICINTERPOLANT":
        log_dir_name += f"_Trev_{args.T_rev}_eps_{args.t_eps}"
    elif ode_class.__name__ == "SCHRODINGERBRIDGE":
        log_dir_name += f"_sigma_{args.sigma}_Trev_{args.T_rev}"
    
    return TensorBoardLogger(
        save_dir="logs",
        name=log_dir_name,
        version=f"run_{timestamp}"
    )

def configure_checkpoints(log_dir_name):
    """配置模型检查点回调"""
    checkpoint_dir = f"logs/{log_dir_name}/checkpoints"
    
    return [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='epoch={epoch}-last',
            save_last=True,
            save_top_k=1
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='epoch={epoch}-pesq={pesq:.2f}',
            monitor='pesq',
            mode='max',
            save_top_k=2
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='epoch={epoch}-si_sdr={si_sdr:.2f}',
            monitor='si_sdr',
            mode='max',
            save_top_k=2
        )
    ]

if __name__ == '__main__':
    # 基础参数解析
    base_parser = ArgumentParser(add_help=False)
    parser = ArgumentParser()
    for parser_ in (base_parser, parser):
        parser_.add_argument("--backbone", type=str, 
                           choices=BackboneRegistry.get_all_names(), 
                           default="ncsnpp")
        parser_.add_argument("--ode", type=str,
                           choices=ODERegistry.get_all_names(),
                           default="otflow")
    
    # 动态添加各模块参数
    temp_args, _ = base_parser.parse_known_args()
    backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
    ode_class = ODERegistry.get_by_name(temp_args.ode)
    
    parser = pl.Trainer.add_argparse_args(parser)
    VFModel.add_argparse_args(parser.add_argument_group("VFModel"))
    ode_class.add_argparse_args(parser.add_argument_group("ODE"))
    backbone_cls.add_argparse_args(parser.add_argument_group("Backbone"))
    SpecsDataModule.add_argparse_args(parser.add_argument_group("DataModule"))

    # 解析参数
    args = parser.parse_args()
    arg_groups = get_argparse_groups(parser)
    dataset_name = os.path.basename(os.path.normpath(args.base_dir))

    # 初始化模型
    model = VFModel(
        backbone=args.backbone,
        ode=args.ode,
        data_module_cls=SpecsDataModule,
        **{
            **vars(arg_groups['VFModel']),
            **vars(arg_groups['ODE']),
            **vars(arg_groups['Backbone']),
            **vars(arg_groups['DataModule'])
        }
    )

    # 配置日志系统
    logger = configure_tensorboard_logger(args, ode_class, dataset_name)
    callbacks = configure_checkpoints(logger.name)

    # 初始化训练器
    trainer = pl.Trainer.from_argparse_args(
        arg_groups['pl.Trainer'],
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator='gpu',
        devices=[1],  # 使用指定的GPU设备
        logger=logger,
        callbacks=callbacks,
        max_epochs=300,
        log_every_n_steps=10,
        num_sanity_val_steps=0
    )

    # 开始训练
    trainer.fit(model)