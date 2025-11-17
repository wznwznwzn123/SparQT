import numpy as np

import torch
import torch.nn as nn
from einops import repeat
from fairscale.nn import checkpoint_wrapper
import torch.nn.functional as F
import math



class Sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self:
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def mlp(num_channels: int):
    return Sequential(
                        nn.LayerNorm(num_channels),
                        nn.Linear(num_channels, num_channels),
                        nn.GELU(),
                        nn.Linear(num_channels, num_channels),
                    )


def cross_attention_layer( num_q_channels: int, 
                           num_kv_channels: int, 
                           num_heads: int, 
                           dropout: float, 
                           activation_checkpoint: bool = False):
    
    layer = Sequential(
        Residual(CrossAttention(num_q_channels, num_kv_channels, num_heads, dropout), dropout),
        Residual(mlp(num_q_channels), dropout),
    )
    return layer if not activation_checkpoint else checkpoint_wrapper(layer)


def self_attention_layer(num_channels: int, 
                         num_heads: int, 
                         dropout: float, 
                         activation_checkpoint: bool = False):
    
    layer = Sequential(
        Residual(SelfAttention(num_channels, num_heads, dropout), dropout), 
        Residual(mlp(num_channels), dropout)
    )
    return layer if not activation_checkpoint else checkpoint_wrapper(layer)


def self_attention_block(num_layers: int, 
                         num_channels: int, 
                         num_heads: int, 
                         dropout: float, 
                         activation_checkpoint: bool = False
                        ):
    
    layers = [self_attention_layer(
                             num_channels, 
                             num_heads, 
                             dropout, 
                             activation_checkpoint) for _ in range(num_layers)]
    
    return Sequential(*layers)


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return self.dropout(x) + args[0]


class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_q_channels: int, 
                 num_kv_channels: int, 
                 num_heads: int, 
                 dropout: float):
        
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=num_q_channels,
            num_heads=num_heads,
            kdim=num_kv_channels,
            vdim=num_kv_channels,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        return self.attention(x_q, x_kv, x_kv, key_padding_mask=pad_mask, attn_mask=attn_mask)[0]


class CrossAttention(nn.Module):
    
    def __init__(self, 
                 num_q_channels: int, 
                 num_kv_channels: int, 
                 num_heads: int, 
                 dropout: float):
        
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_channels)
        self.kv_norm = nn.LayerNorm(num_kv_channels)
        self.attention = MultiHeadAttention(
                                             num_q_channels=num_q_channels, 
                                             num_kv_channels=num_kv_channels, 
                                             num_heads=num_heads, 
                                             dropout=dropout
                                             )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        x_q = self.q_norm(x_q)
        x_kv = self.kv_norm(x_kv)
        return self.attention(x_q, x_kv, pad_mask=pad_mask, attn_mask=attn_mask)


class SelfAttention(nn.Module):
    def __init__(self, num_channels: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
                                             num_q_channels=num_channels, 
                                             num_kv_channels=num_channels, 
                                             num_heads=num_heads, 
                                             dropout=dropout
                                             )

    def forward(self, x, pad_mask=None, attn_mask=None):
        x = self.norm(x)
        return self.attention(x, x, pad_mask=pad_mask, attn_mask=attn_mask)


class Encoder(nn.Module):
    
    def __init__(
        self,
        input_ch,
        preproc_ch,
        
        num_latents: int,
        num_latent_channels: int,
        num_layers: int = 3,
        num_cross_attention_heads: int = 4,
        num_self_attention_heads: int = 4,
        num_self_attention_layers_per_block: int = 6,
        dropout: float = 0.0,
        activation_checkpoint: bool = False,
    ):
        
        super().__init__()

        self.num_layers = num_layers
        if preproc_ch:
            self.preproc = nn.Linear(input_ch, preproc_ch)
        else:
            self.preproc = None
            preproc_ch   = input_ch
            
        def create_layer():
            return Sequential(
                cross_attention_layer(
                    num_q_channels=num_latent_channels,
                    num_kv_channels=preproc_ch, 
                    num_heads=num_cross_attention_heads,
                    dropout=dropout,
                    activation_checkpoint=activation_checkpoint,
                ),
                self_attention_block(
                    num_layers=num_self_attention_layers_per_block,
                    num_channels=num_latent_channels,
                    num_heads=num_self_attention_heads,
                    dropout=dropout,
                    activation_checkpoint=activation_checkpoint,
                ),
            )

        self.layer_1 = create_layer()

        if num_layers > 1:
            self.layer_n = create_layer()
        self.latent = nn.Parameter(torch.empty(num_latents, num_latent_channels))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x, pad_mask=None):
        b, *_ = x.shape
        
        if self.preproc:
            x = self.preproc(x)

        x_latent = repeat(self.latent, "... -> b ...", b=b)
        x_latent = self.layer_1(x_latent, x, pad_mask)


        for i in range(self.num_layers - 1):
            x_latent = self.layer_n(x_latent, x, pad_mask)

        return x_latent


class Decoder(nn.Module):
    def __init__(
        self,
        ff_channels: int,
        preproc_ch,
        num_latent_channels: int,
        latent_size,
        num_output_channels,
        num_cross_attention_heads: int = 4,
        dropout: float = 0.0,
        activation_checkpoint: bool = False,
    ):
        
        super().__init__()
        q_chan = ff_channels + num_latent_channels
        if preproc_ch:
            q_in = preproc_ch
        else:
            q_in = q_chan


        self.postproc = nn.Linear(q_in, num_output_channels)
        
        if preproc_ch:
            self.preproc = nn.Linear(q_chan, preproc_ch)
        else:
            self.preproc = None
        
        self.cross_attention = cross_attention_layer(
                                         num_q_channels=q_in,
                                         num_kv_channels=num_latent_channels,
                                         num_heads=num_cross_attention_heads,
                                         dropout=dropout,
                                         activation_checkpoint=activation_checkpoint,
                                     )

        self.output = nn.Parameter(torch.empty(latent_size,num_latent_channels))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.output.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x, coords):
        b, *_ = x.shape

        output = repeat(self.output, "... -> b ...", b=b)
        output = torch.repeat_interleave(output, coords.shape[1], axis=1)
        
        output = torch.cat([coords,output], axis=-1) 
        
        if self.preproc:
            output = self.preproc(output)
            
        output = self.cross_attention(output, x)
        return self.postproc(output)


class Residual_vq(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual_vq, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        
        layers = []
        for _ in range(self._num_residual_layers):
            layers.append(Residual_vq(in_channels, num_hiddens, num_residual_hiddens))
        self._layers = nn.ModuleList(layers)
    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

    


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        input_shape = inputs.shape
        
        flat_input = inputs.view(-1, self._embedding_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized, perplexity, encoding_indices.view(input_shape[0], -1)
    
    def get_z_q(self, encoding_indices):
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=encoding_indices.device)
        encodings.scatter_(1, encoding_indices, 1)
        z_q = torch.matmul(encodings, self._embedding.weight).view(encoding_indices.shape[0],4,4, self._embedding_dim)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q
    
class LatentVQVAE_Conv2D_Residual(nn.Module):
    def __init__(self, input_dim=48, num_embeddings=1024, embedding_dim=512, commitment_cost=0.25, num_hiddens=128,
                 num_residual_layers=2, 
                 num_residual_hiddens=32):
        super().__init__()
      
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, num_hiddens//2, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(num_hiddens//2, num_hiddens, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1), 
            ResidualStack(in_channels=num_hiddens, num_hiddens=num_hiddens, 
                          num_residual_layers=num_residual_layers, 
                          num_residual_hiddens=num_residual_hiddens),
            nn.Conv2d(num_hiddens, self.embedding_dim, kernel_size=1, stride=1), 
        )
      
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost) 
        self.decoder = nn.Sequential(
            nn.Conv2d(self.embedding_dim, num_hiddens, kernel_size=3, stride=1, padding=1),
            
            ResidualStack(in_channels=num_hiddens, num_hiddens=num_hiddens,
                          num_residual_layers=num_residual_layers,
                          num_residual_hiddens=num_residual_hiddens),
            
            nn.ConvTranspose2d(num_hiddens, num_hiddens//2, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(True),
            
            nn.ConvTranspose2d(num_hiddens//2, input_dim, kernel_size=4, stride=2, padding=1), 

        )
        print(f"Initialized Residual VQ-VAE with {num_embeddings} codes, dim {embedding_dim}, "
              f"{num_residual_layers} residual layers.")
        
    def forward(self, latents):
        x = latents.permute(0, 2, 1).contiguous()
        
        B, C_enc, L = x.shape
        H_reshape = W_rehsape = int(math.sqrt(L))
        x = x.view(B, C_enc, H_reshape, W_rehsape)
        
        z_e = self.encoder(x)
        
        B, C, H, W = z_e.shape
        z_e_for_vq = z_e.permute(0, 2, 3, 1).contiguous()
        
        vq_loss, z_q, perplexity, indices = self.vq(z_e_for_vq)
        
        z_q_for_decoder = z_q.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        reconstructed_x = self.decoder(z_q_for_decoder)
        
        reconstructed_latents = reconstructed_x.view(B, C_enc, L).permute(0, 2, 1).contiguous()
        
        return reconstructed_latents, vq_loss, z_q, indices, perplexity
    
    def get_reconstructed_latents_from_indices(self, indices):
        
        B, T, N = indices.shape
        flat_indices = indices.view(B * T, N)
        
        quantized = self.vq._embedding(flat_indices)
        
        quantized = quantized.view(B * T, 4, 4, self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        decoded = self.decoder(quantized)
        
        decoded = decoded.view(B * T, 48, 256)
        decoded = decoded.permute(0, 2, 1).contiguous()
        
        reconstructed_latents = decoded.view(B, T, 256, 48)
        return reconstructed_latents
    
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
    def forward(self, x_for_shape_info):
        batch_size = x_for_shape_info.size(0)
        seq_len = x_for_shape_info.size(1)
        device = x_for_shape_info.device
        
        pe = torch.zeros(1, seq_len, self.d_model, device=device)
        
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / self.d_model))
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        return pe.repeat(batch_size, 1, 1)


class TimeValueEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)
    def forward(self, query_times: torch.Tensor):
        
        position = query_times.unsqueeze(-1)
        pe = torch.zeros(position.size(0), position.size(1), self.d_model, device=position.device)
        pe[:, :, 0::2] = torch.sin(position * self.div_term)
        pe[:, :, 1::2] = torch.cos(position * self.div_term)
        
        return pe
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            num_q_channels=d_model,
            num_kv_channels=d_model,
            num_heads=nhead,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.GELU()
        
    def forward(self, x, tgt_mask=None, tgt_key_padding_mask=None):
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, pad_mask=tgt_key_padding_mask, attn_mask=tgt_mask)
        x = self.dropout1(x)
        x = residual + x
        
        residual = x
        x = self.norm2(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = residual + x
        
        return x

class TimeTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, max_seq_len=2048):
        super().__init__()
        self.pos_enc = TimeValueEncoding(d_model)
        
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model*4
            ) for _ in range(num_layers)
        ])

        self.register_buffer('causal_mask', None)


    def forward(self, latent_sequence, query_times):
        B, T, _ = latent_sequence.shape
        device = latent_sequence.device
        
        time_embeds = self.pos_enc(query_times)
        
        x = latent_sequence + time_embeds
        
        if self.causal_mask is None or self.causal_mask.size(0) != T:
            self.causal_mask = self._generate_square_subsequent_mask(T, device)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(
                x,
                tgt_mask=self.causal_mask
            )
        
        return x
    
    def _generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
        return mask
    

class RevIN(nn.Module):
    def __init__(self, eps=1e-5, affine=False):
        super(RevIN, self).__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, mode: str, stats=None):
        if mode == 'norm':
            mean, stdev = self._get_statistics(x)
            x_norm = self._normalize(x, mean, stdev)
            return x_norm, (mean, stdev)
        elif mode == 'denorm':
            if stats is None:
                raise ValueError("Denormalization requires statistics (mean, stdev).")
            mean, stdev = stats
            x_denorm = self._denormalize(x, mean, stdev)
            return x_denorm
        else:
            raise NotImplementedError(f"Mode '{mode}' is not implemented.")

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.dim() - 1))
        
        mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        return mean, stdev

    def _normalize(self, x, mean, stdev):
        x = x - mean
        x = x / stdev
        if self.affine:
            x = x * self.weight + self.bias
        return x

    def _denormalize(self, x, mean, stdev):
        if self.affine:
            x = (x - self.bias) / (self.weight + self.eps)
        x = x * stdev
        x = x + mean
        return x