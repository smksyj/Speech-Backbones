# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from model.utils import fix_len_compatibility


# data parameters
train_filelist_path = 'resources/park_eunbo_filelists/metadata_train_210901.txt'
valid_filelist_path = 'resources/park_eunbo_filelists/metadata_val_210901.txt'
test_filelist_path = 'resources/park_eunbo_filelists/metadata_val_210901.txt'
# cmudict_path = 'resources/cmu_dictionary'
n_feats = 128 # mel-spectrogram dim
# add_blank = True
checkpoint_path = 'logs/son/grad_1000.pt'

# encoder parameters
n_enc_channels = 192
filter_channels = 768
filter_channels_dp = 256
n_enc_layers = 6
enc_kernel = 3
enc_dropout = 0.1
n_heads = 2
window_size = 4

# decoder parameters
dec_dim = 64
beta_min = 0.05
beta_max = 20.0
pe_scale = 1000  # 1 for `grad-tts-old.pt` checkpoint

# training parameters
log_dir = 'logs/park_eunbo'
test_size = 4
n_epochs = 10000
batch_size = 30
learning_rate = 1e-4
seed = 37
save_every = 20
out_size = fix_len_compatibility(2*22050//256)
