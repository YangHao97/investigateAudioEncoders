from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from transformers import Wav2Vec2Processor, Wav2Vec2Model, WavLMModel
from s3prl.upstream.interfaces import Featurizer
from s3prl.downstream.asr.expert import DownstreamExpert
#from s3prl.downstream.speech_commands.expert import DownstreamExpert
import yaml
from torch.utils.data import DistributedSampler
from tqdm import tqdm
import random
import numpy as np
import copy
import torchaudio
from pytorch_metric_learning import losses
from s3prl.upstream.wav2vec2_hug.whisper import whisper

SAMPLE_RATE = 16000

################################
# Use Whisper
################################
# class UpstreamExpert(nn.Module):
#     def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
#         super().__init__()

#         #self.processor = Wav2Vec2Processor.from_pretrained(ckpt)
#         self.model = whisper.load_model("small.en")

#     def get_downsample_rates(self, key: str) -> int:
#         return 320

#     def forward(self, wavs: List[Tensor]):
#         device = wavs[0].device
#         wavs_pad = [whisper.pad_or_trim(wav) for wav in wavs]
#         wavs_mel = [whisper.log_mel_spectrogram(wav) for wav in wavs_pad]
#         wavs_input = torch.stack(wavs_mel)
#         outputs = self.model(wavs_input)
#         return {
#             "last_hidden_state": outputs,
#             "hidden_states": None,
#         }

################################
# Use WavLM
################################
# class UpstreamExpert(nn.Module):
#     def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
#         super().__init__()
#         self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-lv60')
#         self.model = WavLMModel.from_pretrained('microsoft/wavlm-large')
#
#
#     def get_downsample_rates(self, key: str) -> int:
#         return 320
#
#
#     def forward(self, wavs: List[Tensor]):
#         device = wavs[0].device
#         processor_outputs = self.processor(
#             [wav.cpu().numpy() for wav in wavs],
#             return_tensors="pt",
#             sampling_rate=SAMPLE_RATE,
#             padding="longest",
#         )
#         attention_mask = processor_outputs.get("attention_mask", None)
#         if isinstance(attention_mask, torch.Tensor):
#             attention_mask = attention_mask.to(device)
#         model_outputs = self.model(
#             processor_outputs.input_values.to(device),
#             attention_mask=attention_mask,
#             output_hidden_states=True,
#         )
#         return {
#             "last_hidden_state": model_outputs.last_hidden_state,
#             "hidden_states": model_outputs.hidden_states,
#         }


################################
# Use Wav2vec2
################################
# class UpstreamExpert(nn.Module):
#     def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
#         super().__init__()
#
#         self.processor = Wav2Vec2Processor.from_pretrained(ckpt)
#         self.model = Wav2Vec2Model.from_pretrained(ckpt)
#
#     def get_downsample_rates(self, key: str) -> int:
#         return 320
#
#
#     def forward(self, wavs: List[Tensor]):
#         device = wavs[0].device
#         processor_outputs = self.processor(
#             [wav.cpu().numpy() for wav in wavs],
#             return_tensors="pt",
#             sampling_rate=SAMPLE_RATE,
#             padding="longest",
#         )
#         attention_mask = processor_outputs.get("attention_mask", None)
#         if isinstance(attention_mask, torch.Tensor):
#             attention_mask = attention_mask.to(device)
#         model_outputs = self.model(
#             processor_outputs.input_values.to(device),
#             attention_mask=attention_mask,
#             output_hidden_states=True,
#         )
#         return {
#             "last_hidden_state": model_outputs.last_hidden_state,
#             "hidden_states": model_outputs.hidden_states,
#         }
