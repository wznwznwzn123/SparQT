# Decoupled Spatiotemporal Forecasting from Extreme Sparse Observations via Quantized Latent Space

## Abstract:
Predicting spatiotemporal fields governed by partial differential equations (PDEs) from sparse sensor data is a critical and long-standing challenge in science and engineering. Recent deep learning approaches, particularly neural operators, have shown considerable promise in solving PDEs. However, their performance degrades significantly in the demanding regime of extreme sparsity, characterized by spatial sensor coverage of less than 1% and limited temporal observations. To overcome this limitation, we propose a novel framework that decouples the task into two stages: spatial reconstruction and temporal extrapolation. In the first stage, rather than reconstructing the high-dimensional physical field directly, our model learns to reconstruct the complete latent features from sparse observations—features that would otherwise be extracted from a dense field. This process is stabilized by a Vector Quantization (VQ) bottleneck, which discretizes the latent space. In the second stage, a decoder-only Transformer performs temporal extrapolation by autoregressively predicting the future sequence of these discrete latent indices. This design inherently allows the model to generalize to new initial conditions and varying forecast horizons, akin to standard autoregressive models. We validate our framework on three challenging benchmarks, achieving state-of-the-art (SOTA) performance under severe sparsity constraints. Furthermore, we introduce a challenging benchmark dataset based on fire dynamics simulations. On this benchmark, our model successfully forecasts the field's evolution 30 frames into the future from a single timeframe with less than 0.1% spatial observations—a result that pushes well beyond the capabilities of existing methods.

# Usage

## Stage1

### sparse_encoder

```python 
python train.py --gpu 0,1 --data cylinder --num_sensors 20 --cons False --seed 123 --enc_preproc 48 --dec_num_latent_channels 48 --enc_num_latent_channels 48 --num_latents 256 --dec_preproc_ch 48 --test False --patience 300 --batch_pixel 2048 --lr 0.0001 --time_transformer_layers 12 --batch_size 1 --freeze_start_on_0 0 --max_X 0 --num_layers 6 --training_stage stage1 --time_sub 1 --batch_frames 64
```


### semi_sparse_encoder

```python 
python train.py --gpu 0,1 --data cylinder --num_sensors 300 --cons False --seed 123 --enc_preproc 48 --dec_num_latent_channels 48 --enc_num_latent_channels 48 --num_latents 256 --dec_preproc_ch 48 --test False --patience 300 --batch_pixel 2048 --lr 0.0001 --time_transformer_layers 12 --batch_size 1 --freeze_start_on_0 0 --max_X 0 --num_layers 6 --training_stage stage1 --time_sub 1 --batch_frames 64
```

## Stage2

Let's say we have the checkpoint of the sprase encoder saved in `logs/2025-07-08_10-25/checkpoints/` and the checkpoint of the sprase encoder saved in `logs/2025-07-08_10-29/checkpoints/`

```python
python train.py --gpu 4,5,6,7 --data cylinder --num_sensors 20 --cons False --seed 123 --enc_preproc 48 --dec_num_latent_channels 48 --enc_num_latent_channels 48 --num_latents 256 --dec_preproc_ch 48 --test False --patience 3000 --batch_pixel 2048 --lr 0.0005 --time_transformer_layers 12 --batch_size 1 --freeze_start_on_0 0 --max_X 0 --num_layers 6 --training_stage stage2 --load_model True --stage2_frames 21 --time_sub 1 --num_vq_embeddings 256 --vq_embedding_dim 512 --batch_frames 128 --model_name 07-08_10-25 --num_hiddens 2048 --semi_sparse True --semi_model_name 07-08_10-29 --load_stage2 True
```

## Stage3

Let's say we have the checkpoint of the Stage2 saved in `logs/2025-07-08_10-30/checkpoints/`

```python
python train.py --gpu 0 --data cylinder --num_sensors 20 --cons False --seed 123 --enc_preproc 48 --dec_num_latent_channels 48 --enc_num_latent_channels 48 --num_latents 256 --dec_preproc_ch 48 --test False --patience 3000 --batch_pixel 2048 --lr 0.0001 --time_transformer_layers 12 --batch_size 1 --freeze_start_on_0 0 --max_X 0 --num_layers 6 --training_stage stage3 --load_model True --load_stage2 True --stage2_frames 21 --time_sub 1 --num_vq_embeddings 256 --vq_embedding_dim 512 --batch_frames 128 --model_name 07-08_10-30 --num_hiddens 4096
```

### Training parameters

- `data_name`: str.
Name of the dataset to be used for training.
- `num_sensors`: int.
number of sensors to train with
- `gpu_device`: int.
GPU to train on. MultiGPU support coming soon.
- `training_frames`: int.
Number of frames (time steps) to train the model with.
- `seed`: int.
If specified, it uses a seed to pick up sensors (if locations not specified) and frames.
- `consecutive_train`: bool.
Whether to use consecutive frames to train or chosen at random.
- `batch_frames`: int.
Number of frames per batch.
- `batch_pixels`: int.
Number of pixels per batch.
- `lr`: float.
Learning rate
- `accum_grads`: int.
Number of batches to accumulate to perform an optimizer step. 


### Model parameters

- `space_bands`: int.
Number of sine-cosine frequencies
- `enc_preproc_ch`: int.
Size of the linear layer that processes the inputs (sensor value+positons)
- `num_latents`: int.
Sequence size of the Q_in array.
- `enc_num_latent_channels`: int.
Channel dimension of the Q_in array.
- `num_layers`: int.
Number of model layers (depth).
- `num_cross_attention_heads`: int.
Number of processsing attention heads.
- `num_self_attention_layers_per_block`: int.
Number of self processing layers in each block.
- `dec_preproc_ch`: int.
Size of the linear layer that processes the latent space sent to the decoder. This can act as a bolttleneck and reduce significantly the number of parameters.
- `dec_num_latent_channels`: int.
Number of channels in the decoder.