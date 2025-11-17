import os
import argparse
import torch

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    parser = argparse.ArgumentParser(description="Senseiver")
    parser.add_argument("--data_name", default=None, type=str)
    parser.add_argument("--test_data_name", default="density.npy", type=str)
    parser.add_argument("--num_sensors", default=8, type=int)
    parser.add_argument("--gpu_device", default='0', type=str)
    parser.add_argument("--training_frames", default=100, type=int)
    parser.add_argument("--consecutive_train", default=False, type=str2bool)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--batch_frames", default=1, type=int)
    parser.add_argument("--pred_steps", default=10, type=int)
    parser.add_argument("--batch_pixels", default=2048, type=int)
    parser.add_argument("--internal_batch_size", default=1, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--accum_grads", default=1, type=int)
    parser.add_argument("--testing_frames", default=0, type=int)
    parser.add_argument("--test_start_frame", default=0, type=int)
    parser.add_argument("--patience", default=100, type=int)
    parser.add_argument("--freeze_start_on_0", default=-1 , type=int)
    parser.add_argument("--max_X", default=0, type=int)
    parser.add_argument("--split_dataloader", default=False, type=str2bool)
    parser.add_argument("--stage2_frames", default=21, type=int)
    parser.add_argument("--space_bands", default=32, type=int)
    parser.add_argument("--stage1_batch_frames", default=1, type=int)
    parser.add_argument("--scheduled_sampling", default=False, type=str2bool)
    parser.add_argument("--time_sub", default=1, type=int)
    parser.add_argument("--load_stage2", default=False, type=str2bool)
    parser.add_argument("--load_model", default=False, type=str2bool)
    parser.add_argument("--load_model_num", default=None, type=int)
    parser.add_argument("--test", default=False, type=str2bool)
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--enc_preproc_ch", default=64, type=int)
    parser.add_argument("--num_latents", default=4, type=int)
    parser.add_argument("--enc_num_latent_channels", default=16, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--num_cross_attention_heads", default=2, type=int)
    parser.add_argument("--enc_num_self_attention_heads", default=2, type=int)
    parser.add_argument("--num_self_attention_layers_per_block", default=3, type=int)
    parser.add_argument("--dropout", default=0.00, type=float)
    parser.add_argument("--num_vq_embeddings", default=1024, type=int)
    parser.add_argument("--vq_embedding_dim", default=256, type=int)
    parser.add_argument("--commitment_cost", default=0.25, type=float)
    parser.add_argument("--num_hiddens", default=128, type=int)
    parser.add_argument("--semi_sparse", default=False, type=str2bool)
    parser.add_argument("--semi_model_name", default=None, type=str)
    parser.add_argument("--dec_preproc_ch", default=None, type=int)
    parser.add_argument("--dec_num_latent_channels", default=16, type=int)
    parser.add_argument("--dec_num_cross_attention_heads", default=1, type=int)
    parser.add_argument("--time_transformer_heads", default=8, type=int)
    parser.add_argument("--time_transformer_layers", default=6, type=int)
    parser.add_argument("--time_bands", default=8, type=float)
    parser.add_argument("--training_stage", default='stage1', type=str)
    args = parser.parse_args()
    test_data_path = args.test_data_name
    if not os.path.isabs(test_data_path):
        test_data_path = os.path.join("Data", "Fire", test_data_path)
    if torch.cuda.is_available():
        accelerator = "gpu"
        gpus = args.gpu_device
    else:
        accelerator = "cpu"
        gpus = None
    data_config = dict(
        data_name=args.data_name,
        test_data_name=test_data_path,
        num_sensors=args.num_sensors,
        gpu_device=None if accelerator == "cpu" else gpus,
        accelerator=accelerator,
        training_frames=args.training_frames,
        consecutive_train=args.consecutive_train,
        seed=args.seed,
        batch_frames=args.batch_frames,
        batch_pixels=args.batch_pixels,
        lr=args.lr,
        accum_grads=args.accum_grads,
        test=args.test,
        space_bands=args.space_bands,
        testing_frames=args.testing_frames,
        load_model=args.load_model,
        test_start_frame=args.test_start_frame,
        patience=args.patience,
        batch_size=args.batch_size,
        pred_steps=args.pred_steps,
        internal_batch_size=args.internal_batch_size,
        freeze_start_on_0=args.freeze_start_on_0,
        max_X=args.max_X,
        stage1_batch_frames=args.stage1_batch_frames,
        split_dataloader=args.split_dataloader,
        stage2_frames=args.stage2_frames,
        scheduled_sampling=args.scheduled_sampling,
        time_sub=args.time_sub,
        model_name=args.model_name,
        load_stage2=args.load_stage2
    )
    encoder_config = dict(
        load_model_num=args.load_model_num,
        enc_preproc_ch=args.enc_preproc_ch,
        num_latents=args.num_latents,
        enc_num_latent_channels=args.enc_num_latent_channels,
        num_layers=args.num_layers,
        num_cross_attention_heads=args.num_cross_attention_heads,
        enc_num_self_attention_heads=args.enc_num_self_attention_heads,
        num_self_attention_layers_per_block=args.num_self_attention_layers_per_block,
        dropout=args.dropout,
    )
    vq_config = dict(
        num_vq_embeddings=args.num_vq_embeddings,
        vq_embedding_dim=args.vq_embedding_dim,
        commitment_cost=args.commitment_cost,
        num_hiddens=args.num_hiddens,
        semi_sparse=args.semi_sparse,
        semi_model_name=args.semi_model_name,
    )
    decoder_config = dict(
        dec_preproc_ch=args.dec_preproc_ch,
        dec_num_latent_channels=args.dec_num_latent_channels,
        latent_size=1,
        dec_num_cross_attention_heads=args.dec_num_cross_attention_heads,
    )
    time_transformer_config = dict(
        time_transformer_heads=args.time_transformer_heads,
        time_transformer_layers=args.time_transformer_layers,
        time_bands=args.time_bands,
        training_stage=args.training_stage,
    )
    return data_config, encoder_config, vq_config, decoder_config, time_transformer_config