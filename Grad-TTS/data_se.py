# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import random
import numpy as np
import torch

from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import parse_filelist, intersperse
from model.utils import fix_len_compatibility
from params import seed as random_seed
from data import TextMelDataset
from KR2Seq import KR2Seq



class TextMelDatasetWithSpeakerEmbs(TextMelDataset):
    def __init__(self, filelist_path, optional_symbols=None):
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.kr2seq = KR2Seq(optional_symbols)
        self.len_symbols = self.kr2seq.n_symbols
        random.seed(random_seed)
        random.shuffle(self.filepaths_and_text)

    def get_pair(self, filepath_and_text):
        filepath, speaker_emb_path, text = filepath_and_text[0], filepath_and_text[1], filepath_and_text[2]
        text = self.get_text(text)
        mel = self.get_mel(filepath)
        speaker_emb = self.get_speaker_emb(speaker_emb_path)
        return (text, mel, speaker_emb)

    def get_mel(self, filepath):
        mel = torch.from_numpy(np.load(filepath)).float()
        return mel

    def get_text(self, text):
        text_norm = self.kr2seq.text_to_ids(text, filter_unknown_symbols=True)
        text_norm = torch.IntTensor(text_norm)
        return text_norm
    
    def get_speaker_emb(self, speaker_emb_path):
#         return torch.from_numpy(np.load(speaker_emb_path))
        return torch.randn(64)

    def __getitem__(self, index):
        text, mel, speaker_emb = self.get_pair(self.filepaths_and_text[index])
        item = {'y': mel, 'x': text, 'speaker_emb': speaker_emb}
        return item

    def __len__(self):
        return len(self.filepaths_and_text)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch
