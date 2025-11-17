import pickle as pk
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from tqdm import tqdm as bar

from model import Encoder, Decoder, VectorQuantizer, TimeTransformer, RevIN, LatentVQVAE_Conv2D_Residual
from positional import PositionalEncoder

from sensor_loc import sensors_3D

from testloss import TestLoss


import numpy as np
import pickle as pk

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from tqdm import tqdm as bar

from model import Encoder, Decoder
from positional import PositionalEncoder

from sensor_loc import sensors_3D

from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import os

class Senseiver(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.stage = self.hparams.get('training_stage', 'stage1')
        self.use_revin = self.hparams.get('use_revin', False)

        if self.use_revin:
            self.revin = RevIN()
        
        pos_encoder_ch = self.hparams.space_bands*len(self.hparams.image_size)*2
        
        self.encoder = Encoder(
            input_ch = self.hparams.im_ch+pos_encoder_ch,
            preproc_ch = self.hparams.enc_preproc_ch,
            num_latents = self.hparams.num_latents,
            num_latent_channels = self.hparams.enc_num_latent_channels,
            num_layers = self.hparams.num_layers,
            num_cross_attention_heads = self.hparams.num_cross_attention_heads,
            num_self_attention_heads = self.hparams.enc_num_self_attention_heads,
            num_self_attention_layers_per_block = self.hparams.num_self_attention_layers_per_block,
            dropout = self.hparams.dropout,
        )
        
        self.decoder = Decoder(
            ff_channels = pos_encoder_ch,
            preproc_ch = self.hparams.dec_preproc_ch,
            num_latent_channels = self.hparams.dec_num_latent_channels,
            latent_size = self.hparams.latent_size,
            num_output_channels = self.hparams.im_ch,
            num_cross_attention_heads = self.hparams.dec_num_cross_attention_heads,
            dropout = self.hparams.dropout,
        )

        if self.hparams.semi_sparse:
            self.semi_sparse_encoder = Encoder(
                input_ch = self.hparams.im_ch+pos_encoder_ch,
                preproc_ch = self.hparams.enc_preproc_ch,
                num_latents = self.hparams.num_latents,
                num_latent_channels = self.hparams.enc_num_latent_channels,
                num_layers = self.hparams.num_layers,
                num_cross_attention_heads = self.hparams.num_cross_attention_heads,
                num_self_attention_heads = self.hparams.enc_num_self_attention_heads,
                num_self_attention_layers_per_block = self.hparams.num_self_attention_layers_per_block,
                dropout = self.hparams.dropout,
            )


        self.vqvae = LatentVQVAE_Conv2D_Residual(
            input_dim=self.hparams.enc_num_latent_channels,
            num_hiddens=self.hparams.num_hiddens,
            num_residual_layers=2,
            num_residual_hiddens=32,
            num_embeddings=self.hparams.num_vq_embeddings,
            embedding_dim=self.hparams.vq_embedding_dim,
            commitment_cost=self.hparams.commitment_cost,
        )

        latent_dim = self.hparams.num_latents * self.hparams.enc_num_latent_channels
        transformer_d_model = self.hparams.get('time_transformer_dim', 2048)
        mlp_hidden_dim = transformer_d_model * 4

        self.input_projection = nn.Sequential(
            nn.Linear(latent_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, transformer_d_model)
        )

        self.time_transformer = TimeTransformer(
            d_model=transformer_d_model,
            nhead=self.hparams.time_transformer_heads,
            num_layers=self.hparams.time_transformer_layers
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(transformer_d_model, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, latent_dim)
        )
        
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.num_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'\nThe model has {self.num_params} params \n')
        
        self._freeze_parameters()
        
    def _freeze_parameters(self):
        if self.stage == 'stage1':
            print("Stage 1: Training encoder-decoder, freezing transformer components")
            for param in self.vqvae.parameters():
                param.requires_grad = False
            for param in self.time_transformer.parameters():
                param.requires_grad = False
            for param in self.input_projection.parameters():
                param.requires_grad = False
            for param in self.output_projection.parameters():
                param.requires_grad = False
                
        elif self.stage == 'stage2':
            print("Stage 2: vqvae, freezing encoder-decoder")
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.time_transformer.parameters():
                param.requires_grad = False
            for param in self.input_projection.parameters():
                param.requires_grad = False
            for param in self.output_projection.parameters():
                param.requires_grad = False
        
        elif self.stage == 'stage3':
            print("Stage 3: time transformer, freezing encoder-decoder")
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.vqvae.parameters():
                param.requires_grad = False
        
    def set_training_stage(self, stage):
        self.stage = stage
        
        for param in self.parameters():
            param.requires_grad = True
            
        self._freeze_parameters()
    
    def encode_sensor_data(self, sensor_values):
        B, T, N, C_sensor = sensor_values.shape
        sv = sensor_values.reshape(B * T, N, C_sensor)
        lat = self.encoder(sv)
        M, D = lat.size(1), lat.size(2)
        lat_flat = lat.reshape(B * T, M * D)
        latents = lat_flat.view(B, T, M * D)
        return latents
    
    def decode_latents(self, latents, query_coords):
        if latents.dim() == 3:
            B, T, latent_dim = latents.shape
            M = self.hparams.num_latents
            D = self.hparams.enc_num_latent_channels
            latents_spatial = latents.reshape(B * T, M, D)
            
            if query_coords.dim() == 4:
                _, _, L, C_coord = query_coords.shape
                query_coords_reshaped = query_coords.reshape(B * T, L, C_coord)
            else:
                query_coords_reshaped = query_coords
                
        else:
            BT, latent_dim = latents.shape
            M = self.hparams.num_latents
            D = self.hparams.enc_num_latent_channels
            latents_spatial = latents.view(BT, M, D)
            query_coords_reshaped = query_coords
        
        preds = self.decoder(latents_spatial, query_coords_reshaped)
        
        if latents.dim() == 3:
            preds = preds.view(B, T, preds.size(1), preds.size(2))
            
        return preds

    def forward(self, sensor_values, query_times, query_coords):
        B = sensor_values.size(0)
        T = query_times.size(1)
        
        sensor_values_expanded = sensor_values.unsqueeze(1)
        t0_latent = self.encode_sensor_data(sensor_values_expanded).squeeze(1)
        t0_latent_proj = self.input_projection(t0_latent)
        
        generated_sequence = [t0_latent_proj]

        for step in range(1, T):
            current_sequence = torch.stack(generated_sequence, dim=1)
            current_times = query_times[:, :step]
            
            transformer_output = self.time_transformer(current_sequence, current_times)
            next_token = transformer_output[:, -1]
            generated_sequence.append(next_token)
            
        
        final_sequence = torch.stack(generated_sequence, dim=1)
        predicted_latents = self.output_projection(final_sequence)
        all_predictions = self.decode_latents(predicted_latents, query_coords)
        
        return all_predictions, final_sequence[:, 1:]


    def training_step(self, batch, batch_idx):
        if self.stage == 'stage1':
            return self._training_step_stage1(batch, batch_idx)
        elif self.stage == 'stage2':
            if self.hparams.get('schedule_sampling', False):
                return self._training_step_stage2_schedule_sampling(batch, batch_idx)
            else:
                return self._training_step_stage2(batch, batch_idx)
        elif self.stage == 'stage3':
            return self._training_step_stage3(batch, batch_idx)
        else:
            raise ValueError(f"Unknown training stage: {self.stage}")

    def _training_step_stage1(self, batch, batch_idx):
        (sensor_values_t0, sensor_values_full, query_times, query_coords, field_values, _, _) = batch
        
        if self.use_revin:
            sensor_vals_only = sensor_values_full[..., :1]
            sensor_pos_codes = sensor_values_full[..., 1:]
            norm_sensor_vals, stats = self.revin(sensor_vals_only, mode='norm')
            norm_sensor_values = torch.cat([norm_sensor_vals, sensor_pos_codes], dim=-1)
        else:
            norm_sensor_values = sensor_values_full

        latents = self.encode_sensor_data(norm_sensor_values)
        predictions_norm = self.decode_latents(latents, query_coords)

        if self.use_revin:
            predictions = self.revin(predictions_norm, mode='denorm', stats=stats)
        else:
            predictions = predictions_norm

        total_loss = F.mse_loss(predictions.reshape(-1), field_values.reshape(-1))
        
        self.log("stage1_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return total_loss
    
    def _training_step_stage2(self, batch, batch_idx):
        (sensor_values_t0, sensor_values_full, query_times, query_coords, field_values, sensors, selected_flat_pixel_indices) = batch

        B, T, N, C = sensor_values_full.shape
        sensor_values_full = sensor_values_full.reshape(B * T, N, C)

        with torch.no_grad():
            all_gt_latents = self.encoder(sensor_values_full)

        recon_latents, vq_loss, z_q, indices, perplexity = self.vqvae(all_gt_latents)

        recon_loss = F.mse_loss(recon_latents.reshape(-1), all_gt_latents.reshape(-1))
        loss = recon_loss + vq_loss

        if self.hparams.semi_sparse:
            semi_sparse_latents = self.semi_sparse_encoder(sensor_values_full)
            semi_sparse_loss = F.mse_loss(semi_sparse_latents.reshape(-1), recon_latents.reshape(-1))
            loss += 0.01 * semi_sparse_loss

        with torch.no_grad():
            recon_field = self.decode_latents(recon_latents.reshape(B, T, -1), query_coords)
            loss_field = F.mse_loss(recon_field.reshape(-1), field_values.reshape(-1))
            
        self.log("recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.log("vq_loss", vq_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.log("loss_field", loss_field, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.log("perplexity", perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.log("stage2_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def _training_step_stage3(self, batch, batch_idx):
        (sensor_values_t0, sensor_values_full, query_times, query_coords, field_values, sensors, selected_flat_pixel_indices) = batch

        B, T, N, C = sensor_values_full.shape

        with torch.no_grad():
            all_gt_latents = self.encoder(sensor_values_full.reshape(B * T, N, C))
            all_gt_latents, vq_loss, z_q, indices, perplexity = self.vqvae(all_gt_latents)

        all_gt_latents_proj = self.input_projection(all_gt_latents.view(B,T,-1))
        predicted_sequence = self.time_transformer(all_gt_latents_proj, query_times)
        
        pred_tokens = predicted_sequence[:, :-1]
        target_tokens = all_gt_latents_proj[:, 1:]
        loss_token = F.mse_loss(pred_tokens, target_tokens)

        predicted_latents_full = self.output_projection(predicted_sequence)
        predicted_latents = predicted_latents_full[:, :-1]
        
        predicted_fields_norm = self.decode_latents(predicted_latents, query_coords[:,1:])
        
        predicted_fields = predicted_fields_norm

        gt_fields_values = field_values[:,1:]
        loss_field = F.mse_loss(predicted_fields.reshape(-1), gt_fields_values.reshape(-1))

        total_loss = F.mse_loss(predicted_latents.reshape(-1), all_gt_latents.reshape(B,T,-1)[:,1:].reshape(-1)) + loss_token * 0.3

        self.log("stage3_loss_token", loss_token,
                on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("stage3_loss_field", loss_field,
                on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("stage3_loss_total", total_loss,
                on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return total_loss

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        lr = self.hparams.get('lr', 1e-4)
        
        if self.stage != 'stage3':
            optimizer = torch.optim.Adam(trainable_params, lr=lr)
        else:
            optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
        
        if self.stage == 'stage1':
            patience = 3
            min_lr = 1e-5
            monitor_loss = "stage1_loss"
        elif self.stage == 'stage2':
            patience = 15
            min_lr = 2e-4
            monitor_loss = "stage2_loss"
        elif self.stage == 'stage3':
            patience = 5
            min_lr = 1e-6
            monitor_loss = "stage3_loss_total"
        else:
            raise ValueError(f"未知的训练阶段: {self.stage}。请从 'stage1', 'stage2', 'stage3' 中选择。")
    
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.8,
            patience=patience,
            verbose=True,
            min_lr=min_lr
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": monitor_loss,
                "interval": "epoch",
                "frequency": 1
            },
        }


    def validation_step(self, batch, batch_idx):
        if self.stage == 'stage1':
            return self._validation_step_stage1(batch, batch_idx)
        elif self.stage == 'stage2':
            return self._validation_step_stage2(batch, batch_idx)
        elif self.stage == 'stage3':
            return self._validation_step_stage3(batch, batch_idx)
        else:
            raise ValueError(f"Unknown training stage: {self.stage}")

    def _validation_step_stage1(self, batch, batch_idx):
        (sensor_values_t0, sensor_values_full, query_times, query_coords, field_values, _, _) = batch
        
        B, T = sensor_values_full.shape[:2]

        if self.use_revin:
            sensor_vals_only = sensor_values_full[..., :1]
            sensor_pos_codes = sensor_values_full[..., 1:]
            norm_sensor_vals, stats = self.revin(sensor_vals_only, mode='norm')
            norm_sensor_values = torch.cat([norm_sensor_vals, sensor_pos_codes], dim=-1)
        else:
            norm_sensor_values = sensor_values_full
        
        total_loss = 0
        total_l2 = 0
        sensor_values = norm_sensor_values

        latents = self.encode_sensor_data(sensor_values)
        predictions_norm = self.decode_latents(latents, query_coords)

        if self.use_revin:
            predictions = self.revin(predictions_norm, mode='denorm', stats=stats)
        else:
            predictions = predictions_norm

        total_loss = F.mse_loss(predictions.reshape(-1), field_values.reshape(-1))
            
        total_l2 = (torch.norm(predictions - field_values, p=2, dim=(1,2)) / torch.norm(field_values, p=2, dim=(1,2))).mean().item()

        self.log("val_stage1_loss", total_loss,
                on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_stage1_l2", total_l2,
                on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return total_loss

    def _validation_step_stage2(self, batch, batch_idx):
        (sensor_values_t0, sensor_values_full, query_times, query_coords, field_values, sensors, selected_flat_pixel_indices) = batch

        B, T, N, C = sensor_values_full.shape
        sensor_values_full = sensor_values_full.reshape(B * T, N, C)

        with torch.no_grad():
            all_gt_latents = self.encoder(sensor_values_full)
            recon_latents, vq_loss, z_q, indices, perplexity = self.vqvae(all_gt_latents)

            recon_loss = F.mse_loss(recon_latents.reshape(-1), all_gt_latents.reshape(-1))
            loss = recon_loss + vq_loss

            if self.hparams.semi_sparse:
                semi_sparse_latents = self.semi_sparse_encoder(sensor_values_full)
                semi_sparse_loss = F.mse_loss(semi_sparse_latents.reshape(-1), recon_latents.reshape(-1))
                loss += 0.01 * semi_sparse_loss

        with torch.no_grad():
            recon_field = self.decode_latents(recon_latents.reshape(B, T, -1), query_coords)
            loss_field = F.mse_loss(recon_field.reshape(-1), field_values.reshape(-1))

        self.log("val_recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_loss_field", loss_field, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_stage2_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
        
        
    def _validation_step_stage3(self, batch, batch_idx):
        (sensor_values_t0, sensor_values_full, query_times, query_coords, field_values, sensors, selected_flat_pixel_indices) = batch
        B, T, N, C = sensor_values_full.shape
        query_times_clone = query_times.clone()

        with torch.no_grad():
            with torch.no_grad():
                all_gt_latents = self.encoder(sensor_values_full.reshape(B * T, N, C))
                all_gt_latents, vq_loss, z_q, indices, perplexity = self.vqvae(all_gt_latents)

            all_gt_latents_proj = self.input_projection(all_gt_latents.view(B,T,-1))
            predicted_sequence = self.time_transformer(all_gt_latents_proj, query_times)
            
            pred_tokens = predicted_sequence[:, :-1]
            target_tokens = all_gt_latents_proj[:, 1:]
            loss_token = F.mse_loss(pred_tokens, target_tokens)

            predicted_latents_full = self.output_projection(predicted_sequence)
            predicted_latents = predicted_latents_full[:, :-1]
            
            predicted_fields_norm = self.decode_latents(predicted_latents, query_coords[:,1:])
            
            predicted_fields = predicted_fields_norm
            
            gt_fields_values = field_values[:,1:]
            loss_field = F.mse_loss(predicted_fields.reshape(-1), gt_fields_values.reshape(-1))

            total_loss = F.mse_loss(predicted_latents.reshape(-1), all_gt_latents.reshape(B,T,-1)[:,1:].reshape(-1)) + loss_token * 0.3

            self.log("val_stage3_loss_token", loss_token,
                    on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("val_stage3_loss_field", loss_field,
                    on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("val_stage3_loss_total", total_loss,
                    on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            autoregressive_predictions, pred_indices = self.forward(
                sensor_values_t0, query_times_clone, query_coords
            )

            loss_field_ar = F.mse_loss(autoregressive_predictions.flatten(), field_values.flatten())

            frame_errors = []
            eval_steps = torch.arange(1,T)
            eval_steps = [s for s in eval_steps if s < T]

            for step in eval_steps:
                pred_slice = autoregressive_predictions[:, :step]
                target_slice = field_values[:, :step]
                error = F.mse_loss(pred_slice.flatten(), target_slice.flatten())
                self.log(f"val_ar_mse_step_{step}", error, on_step=False, on_epoch=True, logger=True, sync_dist=True)
                frame_errors.append({'step': step, 'error': error.item()})

            mask_in = torch.isin(selected_flat_pixel_indices, sensors).squeeze(0)[0]
            mask_out = ~mask_in

            field_values_in = field_values[:, :, mask_in]
            field_values_out = field_values[:, :, mask_out]

            loss_in_s = F.mse_loss(autoregressive_predictions[:, :, mask_in].flatten(), field_values_in.flatten())
            loss_out_s = F.mse_loss(autoregressive_predictions[:, :, mask_out].flatten(), field_values_out.flatten())

            self.log("val_stage3_mse_ar", loss_field_ar, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("val_stage3_ar_mse_in_s", loss_in_s, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log("val_stage3_ar_mse_out_s", loss_out_s, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            
        return loss_field_ar
        
    def load_stage1_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        encoder_state = {k.replace('encoder.', ''): v for k, v in checkpoint['state_dict'].items() 
                        if k.startswith('encoder.')}
        decoder_state = {k.replace('decoder.', ''): v for k, v in checkpoint['state_dict'].items() 
                        if k.startswith('decoder.')}
        
        self.encoder.load_state_dict(encoder_state)
        self.decoder.load_state_dict(decoder_state)
        
        print("Loaded stage 1 checkpoint: encoder and decoder weights")

    def load_stage2_checkpoint(self, checkpoint_path, checkpoint_semi_sparse_path=None):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        encoder_state = {k.replace('encoder.', ''): v for k, v in checkpoint['state_dict'].items() 
                        if k.startswith('encoder.')}
        decoder_state = {k.replace('decoder.', ''): v for k, v in checkpoint['state_dict'].items() 
                        if k.startswith('decoder.')}

        if checkpoint_semi_sparse_path is not None:
            checkpoint_semi_sparse = torch.load(checkpoint_semi_sparse_path, map_location=self.device)
            semi_sparse_encoder_state = {k.replace('encoder.', ''): v for k, v in checkpoint_semi_sparse['state_dict'].items() 
                                        if k.startswith('encoder.')}
            
            self.semi_sparse_encoder.load_state_dict(semi_sparse_encoder_state)

            for params in self.semi_sparse_encoder.parameters():
                params.requires_grad = False

        self.encoder.load_state_dict(encoder_state)
        self.decoder.load_state_dict(decoder_state)
        
        print("Loaded stage 2 checkpoint: encoder, decoder, and vqvae weights")

    def _training_step_stage2_scheduled_sampling(self, batch, batch_idx):
        (sensor_values_t0, sensor_values_full, query_times, query_coords, field_values, sensors, selected_flat_pixel_indices) = batch
        B, T_full = sensor_values_full.shape[:2]
        num_pred_steps = T_full - 1
        if self.use_revin:
            sensor_vals_full_only = sensor_values_full[..., :1]
            sensor_pos_codes_full = sensor_values_full[..., 1:]
            norm_sensor_values_only, stats = self.revin(sensor_vals_full_only, mode='norm')
            norm_sensor_values = torch.cat([norm_sensor_values_only, sensor_pos_codes_full], dim=-1)
            sensor_vals_t0_only = sensor_values_t0[..., :1]
            sensor_pos_codes_t0 = sensor_values_t0[..., 1:]
            norm_sensor_values_t0_only, _ = self.revin(sensor_vals_t0_only, mode='norm', stats=stats)
            norm_sensor_values_t0 = torch.cat([norm_sensor_values_t0_only, sensor_pos_codes_t0], dim=-1)
        else:
            norm_sensor_values = sensor_values_full
            norm_sensor_values_t0 = sensor_values_t0
            stats = None
        with torch.no_grad():
            all_gt_latents = self.encode_sensor_data(norm_sensor_values)
        t0_latent = self.encode_sensor_data(norm_sensor_values_t0.unsqueeze(1))
        
        all_gt_latents_proj = self.input_projection(all_gt_latents)
        t0_latent_proj = self.input_projection(t0_latent)
        
        p = 0.5
        current_input_sequence = t0_latent_proj
        collected_predictions = []
        for t in range(num_pred_steps):
            predicted_sequence_output = self.time_transformer(current_input_sequence, query_times[:, :t+1])
            last_prediction = predicted_sequence_output[:, -1, :].unsqueeze(1)
            collected_predictions.append(last_prediction)
            if torch.rand(1).item() < p:
                next_input = last_prediction
            else:
                next_input = all_gt_latents_proj[:, t+1, :].unsqueeze(1)
            
            current_input_sequence = torch.cat([current_input_sequence, next_input], dim=1)

        predicted_sequence = torch.cat(collected_predictions, dim=1)

        pred_tokens = predicted_sequence
        target_tokens = all_gt_latents_proj[:, 1:]
        loss_token = F.mse_loss(pred_tokens, target_tokens)
        predicted_sequence_for_decode = torch.cat((t0_latent_proj, predicted_sequence), dim=1)
        predicted_latents_full = self.output_projection(predicted_sequence_for_decode)
        predicted_latents = predicted_latents_full
        predicted_fields_norm = self.decode_latents(predicted_latents, query_coords)
        
        if self.use_revin:
            predicted_fields = self.revin(predicted_fields_norm, mode='denorm', stats=stats)
        else:
            predicted_fields = predicted_fields_norm
        
        gt_fields_values = field_values
        loss_field = F.mse_loss(predicted_fields.reshape(-1), gt_fields_values.reshape(-1))
        total_loss = F.mse_loss(predicted_latents.reshape(-1), all_gt_latents.reshape(-1)) + loss_token * 0.3
        self.log("stage2_loss_field", loss_field,
                on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("stage2_loss_total", total_loss,
                on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return total_loss
        

    class LossPlotCallback(Callback):
        def __init__(self, save_dir='loss_plots'):
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
            self.train_losses = {}
            self.val_losses = {}
            
        def on_train_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            
            train_losses = {k: v for k, v in metrics.items() if k.startswith('stage_')}
            for name, value in train_losses.items():
                if name not in self.train_losses:
                    self.train_losses[name] = []
                self.train_losses[name].append(value.cpu().item())
                
            val_losses = {k: v for k, v in metrics.items() if k.startswith('val_')}
            for name, value in val_losses.items():
                if name not in self.val_losses:
                    self.val_losses[name] = []
                self.val_losses[name].append(value.cpu().item())
                
            plt.figure(figsize=(12, 6))
            
            for name, values in self.train_losses.items():
                plt.plot(range(len(values)), values, label=name)
                
            for name, values in self.val_losses.items():
                plt.plot(range(len(values)), values, label=name)
                
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Losses')
            plt.legend()
            plt.grid(True)
            
            plt.savefig(os.path.join(self.save_dir, f'loss_plot_epoch_{trainer.current_epoch}.png'))
            plt.close()