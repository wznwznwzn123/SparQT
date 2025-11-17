import os

import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import torch
import pytz

from datasets import cylinder, NOAA, pipe, plume, porous, Fire_3D, shallow

from sensor_loc import (cylinder_16_sensors,
                        cylinder_8_sensors,
                        cylinder_4BC_sensors,
                        sea_n_sensors,
                        sensors_3D,
                        sensors_3D_custom,
                        shallow_water_n_sensors
                        )


import datetime
from positional import PositionalEncoder


from torch.utils.data import DataLoader, Dataset

from s_parser import parse_args


data_config, encoder_config, vq_config, decoder_config, time_transformer_config = parse_args()


def load_data(dataset_name, num_sensors, seed=123):

    if dataset_name == 'shallow':
        data = shallow()

        x_sens, y_sens = shallow_water_n_sensors(data[0]['data'].squeeze(), num_sensors, seed)

    elif dataset_name == 'cylinder':
        data = cylinder()

        if num_sensors == 16:
            x_sens, y_sens = cylinder_16_sensors()

        x_sens, y_sens = cylinder_8_sensors(data[0]['data'].squeeze(), num_sensors, seed)

        if num_sensors == 4:
            x_sens, y_sens = cylinder_4_sensors()

        if num_sensors == 4444:
            x_sens, y_sens = cylinder_4BC_sensors()

    elif dataset_name == 'sea':
        data = NOAA()
        x_sens, y_sens = sea_n_sensors(data, num_sensors, seed)

    elif dataset_name == 'pipe':
       data = pipe()
       x_sens, y_sens = sea_n_sensors(data, num_sensors, seed)

    elif dataset_name == 'plume':
        data = plume()
        data = data[None, :, :, :, None]
        x_sens, *y_sens = sensors_3D(data, num_sensors, seed)

    elif dataset_name == 'pore':
        data = porous()
        data = data[:, :, :, :, None]
        x_sens, *y_sens = sensors_3D(data, num_sensors, seed)

    elif dataset_name == 'fire':
        data = Fire_3D()

        x_sens, *y_sens = sensors_3D(data[0]['data'], num_sensors, seed)

    else:
        print(f'The dataset_name {dataset_name} was not provided\n')
        print('************WARNING************')
        print('*******************************\n')
        print('Creating a dummy dataset\n')
        print('************WARNING************')
        print('*******************************\n')
        data = np.random.rand(1000, 150, 75, 1)
        x_sens, y_sens = sea_n_sensors(data, num_sensors, seed)

    for i in range(len(data)):
        if data[i]['data'].shape[-1] == 1:
            data[i]['coords'] = torch.as_tensor(data[i]['coords'], dtype=torch.float)
        else:
            data[i]['coords'] = torch.as_tensor(data[i]['coords'], dtype=torch.float)
    return data, x_sens, y_sens


def senseiver_dataloader(data_config, num_workers=0):
    return DataLoader(senseiver_loader(data_config),
                      batch_size=data_config['batch_size'],
                      pin_memory=True,
                      shuffle=True,
                      num_workers=4,
                      persistent_workers=True
                      )



def valloader(data_config, num_workers=0):
    return DataLoader(SenseiverValLoader(data_config),
                      batch_size=data_config['batch_size'],
                      pin_memory=True,
                      shuffle=False,
                      num_workers=4,
                      persistent_workers=True
                      )


def testloader(data_config, num_workers=0):
    return DataLoader(SenseiverTestLoader(data_config),
                      batch_size=data_config['batch_size'],
                      pin_memory=True,
                      shuffle=False,
                      num_workers=4,
                      persistent_workers=True
                      )


class senseiver_loader(Dataset):
    def __init__(self, data_config, mode='train'):
        data_name = data_config['data_name']
        num_sensors = data_config['num_sensors']
        seed = data_config['seed']
        self.training_stage = time_transformer_config['training_stage']

        self.data_name = data_name
        self.mode = mode

        loaded_scenes, x_sens, y_sens = load_data(data_name, num_sensors, seed)

        if seed:
            torch.manual_seed(seed)

        num_scenes = len(loaded_scenes)
        scene_indices = torch.arange(num_scenes)

        if data_name == 'cylinder':
            val_indices = scene_indices[266:272]
            test_indices = scene_indices[250:266]
            train_indices = scene_indices[:250]
        elif data_name == 'shallow':
            val_indices = scene_indices[8:10]
            test_indices = scene_indices[8:10]
            train_indices = scene_indices[:8]
        elif data_name == 'fire':
            val_indices = scene_indices[20:21]
            test_indices = scene_indices[30:32]
            train_indices = scene_indices[:20]

        if mode == 'train':
            self.scenes_data = [loaded_scenes[i] for i in train_indices]
        elif mode == 'val':
            self.scenes_data = [loaded_scenes[i] for i in val_indices]
        elif mode == 'test':
            self.scenes_data = [loaded_scenes[i] for i in test_indices]

        self.num_scenes = len(self.scenes_data)
        data_config['num_scenes'] = self.num_scenes

        print(f'Total scenes: {num_scenes}')

        if not self.scenes_data:
            raise ValueError("No training data available. Check dataset_name and data loading logic.")

        first_scene_data = loaded_scenes[0]['data']
        first_scene_coords = loaded_scenes[0]['coords']

        self.data = first_scene_data[10:40]
        self.coords = first_scene_coords
        np.save("coords_s.npy", self.coords)

        total_frames, *spatial_dims, im_ch = first_scene_data.shape
        image_size = list(spatial_dims)

        print("image size:", image_size)
        print("First scene data shape:", first_scene_data.shape)
        print("First scene coords shape:", first_scene_coords.shape)
        print("First scene data max:", first_scene_data.max())
        print("First scene data min:", first_scene_data.min())

        self.total_frames = total_frames

        data_config['total_frames'] = total_frames
        data_config['image_size'] = image_size
        data_config['im_ch'] = im_ch

        self.starting_frame = data_config['test_start_frame']
        self.batch_frames = data_config['batch_frames']
        self.batch_pixels = data_config['batch_pixels']
        self.internal_batch_size = data_config.get('internal_batch_size', 1)
        self.testing_frames = data_config.get('testing_frames', 0)

        max_start_frame = data_config.get('max_X', self.total_frames // 3)
        self.max_start_frame = max_start_frame

        print(f'Total frames: {self.total_frames}, Batch frames: {self.batch_frames}')
        print(f'Max start frame: {max_start_frame}')

        self.available_starts = []
        for scene_idx in range(self.num_scenes):
            scene_starts = torch.arange(0, total_frames - 1, data_config['stage2_frames'])

            self.available_starts.append(scene_starts)

        self.total_seq_per_scene = len(self.available_starts[0])

        if data_config['freeze_start_on_0'] != -1:
            self.available_starts = [torch.as_tensor([0]) for _ in range(self.num_scenes)]
        if self.data_name == 'fire':
            self.available_starts = [torch.as_tensor([10]) for _ in range(self.num_scenes)]

        self.test_ind = []
        for scene_idx in range(self.num_scenes):
            scene_test_ind = torch.arange(total_frames)[self.starting_frame:self.testing_frames + self.starting_frame]
            self.test_ind.append(scene_test_ind)

        print(x_sens, y_sens)

        sensors = np.zeros(image_size)
        if data_name == 'fire':
            sensors = sensors.transpose(1, 0, 2)
            sensors[x_sens, y_sens[0], y_sens[1]] = 1
        else:
            if len(image_size) == 3:
                if isinstance(y_sens, (list, tuple)) and len(y_sens) >= 2:
                    sensors[x_sens, y_sens[0], y_sens[1]] = 1
            elif len(image_size) == 2:
                sensors[x_sens, y_sens] = 1

        self.sensors, *_ = np.where(sensors.flatten() == 1)

        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)
        sensor_file = f'{output_dir}/sensors_pos.npy'
        np.save(sensor_file, self.sensors)

        self.pos_encodings = PositionalEncoder(first_scene_data.shape[1:], data_config['space_bands'])

        self.indexed_sensors = []
        self.sensor_positions = []
        common_sensor_pos_encoding = self.pos_encodings[self.sensors,]

        for scene_idx in range(self.num_scenes):
            current_scene_data_tensor = self.scenes_data[scene_idx]['data']
            current_scene_coords_tensor = self.scenes_data[scene_idx]['coords']

            scene_flattened_data = current_scene_data_tensor.flatten(start_dim=1, end_dim=-2)
            scene_indexed_sensors = scene_flattened_data[:, self.sensors, :]
            self.indexed_sensors.append(scene_indexed_sensors)
            self.sensor_positions.append(common_sensor_pos_encoding)

            self.scenes_data[scene_idx]['coords_flat'] = current_scene_coords_tensor.flatten(start_dim=0, end_dim=-2)

        self.pix_avail = []
        total_available_pixels = 0

        for scene_idx in range(self.num_scenes):
            current_scene_data_tensor = self.scenes_data[scene_idx]['data']
            scene_flattened = current_scene_data_tensor.flatten(start_dim=1, end_dim=-2)
            if data_name == 'fire':
                ref_frame_idx = 20
                ref_frame_all_channels = scene_flattened[ref_frame_idx, :, 0]
                scene_pix_avail = ref_frame_all_channels.nonzero()[:, 0]
                self.pix_avail.append(scene_pix_avail)
                total_available_pixels += len(scene_pix_avail)
                continue
            else:
                ref_frame_all_channels = scene_flattened[0, :, :]

            is_pixel_nonzero = torch.any(ref_frame_all_channels != 0, dim=-1)

            scene_pix_avail = is_pixel_nonzero.nonzero(as_tuple=False).squeeze(-1)

            self.pix_avail.append(scene_pix_avail)
            total_available_pixels += len(scene_pix_avail)

        total_available_sequences = self.total_seq_per_scene * self.num_scenes if self.total_seq_per_scene > 0 else self.num_scenes

        if self.batch_pixels > 0 and total_available_sequences > 0 and total_available_pixels > 0:
            avg_pixels_per_scene = total_available_pixels / self.num_scenes
            total_data_points = total_available_sequences * avg_pixels_per_scene
            self.num_batches = max(1, int(total_data_points / self.batch_pixels / (self.total_seq_per_scene)))
            if self.training_stage == 'stage1':
                self.num_batches = max(1, int(avg_pixels_per_scene / self.batch_pixels) * self.num_scenes)
        else:
            self.num_batches = 0


        if self.training_stage == 'stage1' or self.training_stage == 'stage2':
            data_tensors_list = [d['data'] for d in self.scenes_data]
            self.scenes_data = torch.cat(data_tensors_list, dim=0)
            self.indexed_sensors = torch.cat(self.indexed_sensors, dim=0)

            total_frames = self.scenes_data.shape[0]

            all_indices = torch.arange(0, total_frames)
            if data_name == 'cylinder':
                segments = all_indices.split(40)
                first_21_indices = [seg[:21][::data_config['time_sub']] for seg in segments]
            elif data_name == 'shallow':
                segments = all_indices.split(160)
                first_21_indices = [seg[:160][::data_config['time_sub']] for seg in segments]
            elif data_name == 'fire':
                segments = all_indices.split(120)
                first_21_indices = [seg[10:40][::data_config['time_sub']] for seg in segments]


            self.train_ind = torch.cat(first_21_indices, dim=0)

            self.training_frames = len(self.train_ind)

            self.num_batches = int(self.scenes_data.shape[1:].numel() * self.training_frames / (
                                                 self.batch_frames * self.batch_pixels))


        data_config['num_batches'] = self.num_batches
        print(f'Total batches: {self.num_batches}')

        self.pt = 0

        if seed:
            torch.manual_seed(datetime.datetime.now().microsecond)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        scene_idx = torch.randint(0, self.num_scenes, (1,)).item()

        if self.training_stage == 'stage1' or self.training_stage == 'stage2':
            frames = self.train_ind[torch.randperm(self.training_frames)][:self.batch_frames]


            sensor_values = self.indexed_sensors[frames, ]
            sensor_values = torch.cat([sensor_values, self.sensor_positions[0][None, ].repeat_interleave(len(frames), axis=0)], axis=-1)

            scene_pix_avail = self.pix_avail[scene_idx]
            num_pixels_to_sample = min(self.batch_pixels, len(scene_pix_avail))

            if len(scene_pix_avail) > 0 and num_pixels_to_sample > 0:
                selected_flat_pixel_indices = scene_pix_avail[torch.randperm(len(scene_pix_avail))[:num_pixels_to_sample]]
            else:
                selected_flat_pixel_indices = torch.empty(0, dtype=torch.long)

            if len(selected_flat_pixel_indices) > 0:
                pixel_pos_encodings = self.pos_encodings[selected_flat_pixel_indices, :]
                query_coords = pixel_pos_encodings.unsqueeze(0).repeat(len(frames), 1, 1)
            else:
                c_coord_pixel_dim = self.pos_encodings.output_dim

            field_values = self.scenes_data.flatten(start_dim=1, end_dim=-2)[frames, ][:, selected_flat_pixel_indices, ]

            return 0, sensor_values, 0, query_coords, field_values, 0, 0


        available_starts_for_scene = self.available_starts[scene_idx]
        if len(available_starts_for_scene) == 0:
            print(available_starts_for_scene)
            raise RuntimeError(f"No available start frames for scene {scene_idx}")

        start_idx_in_available = torch.randint(0, len(available_starts_for_scene), (1,)).item()
        start_frame = available_starts_for_scene[start_idx_in_available].item()

        sequence_length = data_config['stage2_frames']

        known_frames_indices = torch.tensor(start_frame)

        all_frames_indices = torch.arange(start_frame, start_frame + sequence_length)

        all_frames_indices = all_frames_indices[::data_config['time_sub']]

        sensor_data_t0_raw = self.indexed_sensors[scene_idx][known_frames_indices, :, :]
        sensor_pos_encoding = self.sensor_positions[scene_idx]
        sensor_values_t0 = torch.cat([sensor_data_t0_raw, sensor_pos_encoding], dim=-1)

        sensor_data_known_raw = self.indexed_sensors[scene_idx][all_frames_indices, :, :]
        sensor_pos_expanded = sensor_pos_encoding.unsqueeze(0).repeat(len(all_frames_indices), 1, 1)
        sensor_values_full = torch.cat([sensor_data_known_raw, sensor_pos_expanded], dim=-1)

        if self.data_name == 'shallow':
            query_times = all_frames_indices.float() / self.total_frames
        elif self.data_name == 'cylinder':
            query_times = all_frames_indices.float() / 20
        elif self.data_name == 'fire':
            query_times = torch.arange(30).float() / 30
            query_times = query_times[::data_config['time_sub']]

        scene_pix_avail = self.pix_avail[scene_idx]
        num_pixels_to_sample = min(self.batch_pixels, len(scene_pix_avail))

        if len(scene_pix_avail) > 0 and num_pixels_to_sample > 0:
            selected_flat_pixel_indices = scene_pix_avail[torch.randperm(len(scene_pix_avail))[:num_pixels_to_sample]]
        else:
            selected_flat_pixel_indices = torch.empty(0, dtype=torch.long)

        if len(selected_flat_pixel_indices) > 0:
            pixel_pos_encodings = self.pos_encodings[selected_flat_pixel_indices, :]
            query_coords = pixel_pos_encodings.unsqueeze(0).repeat(len(all_frames_indices), 1, 1)
        else:
            c_coord_pixel_dim = self.pos_encodings.output_dim
            query_coords = torch.empty((len(all_frames_indices), 0, c_coord_pixel_dim), dtype=torch.float32)

        current_scene_full_data = self.scenes_data[scene_idx]['data']
        scene_data_flat = current_scene_full_data.flatten(start_dim=1, end_dim=-2)

        if len(selected_flat_pixel_indices) > 0:
            field_values = scene_data_flat[all_frames_indices][:, selected_flat_pixel_indices, :]
        else:
            im_ch = current_scene_full_data.shape[-1]
            field_values = torch.empty((len(all_frames_indices), 0, im_ch), dtype=scene_data_flat.dtype)

        return (sensor_values_t0, sensor_values_full, query_times, query_coords, field_values, self.sensors, selected_flat_pixel_indices)


class SenseiverValLoader(senseiver_loader):
    def __init__(self, data_config):
        super().__init__(data_config, mode='val')


class SenseiverTestLoader(senseiver_loader):
    def __init__(self, data_config):
        super().__init__(data_config, mode='test')