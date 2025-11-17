import numpy as np
import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1,2,3'
import pytz
import time
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from glob import glob as gb
from scipy.spatial.distance import jensenshannon
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from testloss import TestLoss
from pytorch_lightning.callbacks import LearningRateMonitor
logger = TensorBoardLogger('logs/', name='senseiver')

from s_parser import parse_args
from dataloaders import senseiver_dataloader, valloader, testloader, senseiver_dataloader_stage1, valloader_stage1
from network_light import Senseiver, LossPlotCallback

from plot import plot_cs


# arg parser
data_config, encoder_config, vq_config, decoder_config, time_transformer_config = parse_args()

dataloader = senseiver_dataloader(data_config, num_workers=4)
valid_loader = valloader(data_config, num_workers=4)


# instantiate new Senseiver
model = Senseiver(
    **encoder_config,   
    **decoder_config,
    **vq_config,
    **data_config,
    **time_transformer_config,
)


os.environ['CUDA_VISIBLE_DEVICES'] = data_config['gpu_device']

gpu_indices = [int(i) for i in data_config['gpu_device'].split(',')]
# 在PyTorch等多卡环境中，你实际使用的设备ID会从0开始重新编号
# 例如，如果CUDA_VISIBLE_DEVICES='1,6,7'，那么设备'1'在代码中是device:0，设备'6'是device:1
device_idx = list(range(len(gpu_indices)))

model_loc = None
semi_model_loc = None

# load model (if requested)
if data_config['load_model']:
    load_stage2 = data_config['load_stage2']
    
    model_name = data_config['model_name']
    semi_model_name = vq_config['semi_model_name']
    
    # 使用 f-string 动态构建 glob 搜索模式
    # 将 'logs/2025-'、model_name 和 '/checkpoints/*.ckpt' 拼接起来
    glob_pattern = f"logs/2025-{model_name}/checkpoints/*.ckpt"
    glob_pattern_semi = f"logs/2025-{semi_model_name}/checkpoints/*.ckpt"
    
    print(f"将要使用的Glob模式: {glob_pattern}")
    
    # 使用 glob 搜索匹配的文件
    checkpoint_files = gb(glob_pattern)
    checkpoint_files_semi = gb(glob_pattern_semi)
    
    # 检查是否找到了文件
    if checkpoint_files:
        # 获取列表中的第一个文件
        model_loc = checkpoint_files[0]
        print(f"成功找到并加载模型: {model_loc}")
    else:
        # 如果没有找到任何文件，进行提示
        print(f"警告：在路径 {os.path.dirname(glob_pattern)} 下没有找到任何 .ckpt 文件！")
        model_loc = None

    if checkpoint_files_semi:
        # 获取列表中的第一个文件
        semi_model_loc = checkpoint_files_semi[0]
        print(f"成功找到并加载半监督模型: {semi_model_loc}")
    else:
        # 如果没有找到任何文件，进行提示
        print(f"警告：在路径 {os.path.dirname(glob_pattern_semi)} 下没有找到任何 .ckpt 文件！")
        semi_model_loc = None

    print(f'Loading from new path: {model_loc}')
    
    if load_stage2:
        if semi_model_loc is not None:
            model.load_stage2_checkpoint(model_loc, semi_model_loc)
        else:
            model.load_stage2_checkpoint(model_loc)
    else:
        model = Senseiver.load_from_checkpoint(model_loc,
                                        strict=False,
                                       **encoder_config,
                                       **decoder_config,
                                       **vq_config,
                                       **data_config,
                                       **time_transformer_config)
if data_config['test']:
    # 直接指定新的checkpoints路径格式
    test_loader = testloader(data_config, num_workers=4)
    model_loc = gb("logs/2025-07-15_18-45/checkpoints/*.ckpt")[0]  # 修改为你的实际路径
    print(f'Loading from new path: {model_loc}')
    model = Senseiver.load_from_checkpoint(model_loc,
                                        strict=False,
                                       **encoder_config,
                                       **decoder_config,
                                       **vq_config,
                                       **data_config,
                                       **time_transformer_config)

if not data_config['test']:
    timezone = pytz.timezone("Asia/Shanghai") 
    # 获取当前时间并格式化为字符串
    timestamp = datetime.now(timezone).strftime("%Y-%m-%d_%H-%M")

    # 创建基于时间戳的日志文件夹路径
    log_dir = os.path.join('logs', timestamp)

    # 确保日志和检查点目录存在
    os.makedirs(log_dir, exist_ok=True)  # 创建 logs/时间戳 文件夹
    os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)
    # os.makedirs(os.path.join(log_dir, 'loss_plots'), exist_ok=True)

    # 更新 TensorBoardLogger 的路径
    logger = TensorBoardLogger(save_dir=log_dir, name='senseiver')
    # Loss_plot_callback = LossPlotCallback(save_dir=os.path.join(log_dir, 'loss_plots'))

    stage = time_transformer_config['training_stage']

    # 回调函数：ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_recon_loss" if stage == 'stage2' else "val_stage1_loss",
        # monitor="val_stage1_loss",
        filename="train-{epoch:02d}",
        every_n_epochs=50 if time_transformer_config['training_stage'] == 'stage2' else 30,
        save_on_train_epoch_end=False,
        save_top_k=1,
        verbose=True,
        mode="min",
        save_weights_only=True,
        dirpath=os.path.join(log_dir, 'checkpoints')  # 检查点保存路径
    )

    # EarlyStopping 回调
    early_stopping_callback = EarlyStopping(
        monitor="val_stage2_loss" if stage == 'stage2' else "stage1_loss",
        check_finite=False,
        patience=data_config['patience'],
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')



    # 定义 Trainer
    trainer = Trainer(
        logger=logger,
        max_epochs=8888, 
        callbacks=[checkpoint_callback, lr_monitor],
        # callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator=data_config['accelerator'],
        devices=device_idx,
        strategy="ddp",
        accumulate_grad_batches=data_config['accum_grads'],
        log_every_n_steps=data_config['num_batches'] // (data_config['batch_size'] * len(device_idx)),
        check_val_every_n_epoch=5 if time_transformer_config['training_stage'] == 'stage1' else 1,
    )
    
    model_loc = None
    trainer.fit(model, dataloader, val_dataloaders=valid_loader,
            ckpt_path=model_loc
        )
    
# 1 传感器放在z=0
# 2 放宽筛选条件，加上位置约束
# 3 观察训练过程的误差变化
else:
    if data_config['gpu_device']:
        device = data_config['gpu_device'][0]
        model = model.to(f"cuda:{device}")

        model = model.to(f"cuda:{data_config['gpu_device'][0]}")
        # 新增自定义测试帧
        # scene_idx = 0
        # # test_ind = dataloader.dataset.test_ind[scene_idx]
        # dataloader.dataset.data = torch.as_tensor(
        #     dataloader.dataset.data).to(f"cuda:{device}")
        # dataloader.dataset.sensors = torch.as_tensor(
        #     dataloader.dataset.sensors).to(f"cuda:{device}")
        # dataloader.dataset.pos_encodings = torch.as_tensor(
        #     dataloader.dataset.pos_encodings).to(f"cuda:{device}")
        # dataloader.dataset.test_data = torch.as_tensor(
        #     dataloader.dataset.test_data).to(f"cuda:{device}")

    path = model_loc.split('checkpoints')[0]

    def r2_score(y_true, y_pred):
        # 计算总平方和 (TSS)
        mean_y_true = torch.mean(y_true)
        tss = torch.sum((y_true - mean_y_true) ** 2)
        # 计算残差平方和 (RSS)
        rss = torch.sum((y_true - y_pred) ** 2)
        # 计算R²
        r2 = 1 - (rss / tss)
        return r2.item()

    def test_model(model, test_loader):
        """
        测试模型并计算有意义区域的误差
        """
        model.eval()
        with torch.no_grad():
            start_time = time.time()

            # 初始化累积变量
            total_loss_field_ar = 0.0
            total_loss_in_s = 0.0
            total_loss_out_s = 0.0
            total_loss_sum = 0.0
            batch_count = 0

            for batch_idx, batch in enumerate(test_loader):
                batch = [b.to(model.device) for b in batch]
                loss_field_ar, loss_in_s, loss_out_s, total_loss = model._validation_step_stage2(batch, batch_idx)
                
                # 累积损失值
                total_loss_field_ar += loss_field_ar
                total_loss_in_s += loss_in_s
                total_loss_out_s += loss_out_s
                total_loss_sum += total_loss
                batch_count += 1
                
                print(f"Batch {batch_idx}: loss_field_ar={loss_field_ar:.7f}, loss_in_s={loss_in_s:.7f}, loss_out_s={loss_out_s:.7f}, total_loss={total_loss:.7f}")

            # 计算平均损失
            avg_loss_field_ar = total_loss_field_ar / batch_count
            avg_loss_in_s = total_loss_in_s / batch_count
            avg_loss_out_s = total_loss_out_s / batch_count
            avg_total_loss = total_loss_sum / batch_count
            
            print(f"\n=== 平均损失统计 ===")
            print(f"总批次数: {batch_count}")
            print(f"平均 loss_field_ar: {avg_loss_field_ar:.7f}")
            print(f"平均 loss_in_s: {avg_loss_in_s:.7f}")
            print(f"平均 loss_out_s: {avg_loss_out_s:.7f}")
            print(f"平均 total_loss: {avg_total_loss:.7f}")

            end_time = time.time()  # 记录推理结束时间
            inference_time = end_time - start_time
            print(f"Inference time: {inference_time:.4f} seconds")

            return 
