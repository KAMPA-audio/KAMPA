# -*- coding: utf-8 -*-
"""세 모델의 공통 뼈대 — 파형 [B, T](16 kHz) -> frontend(mel) -> (학습 중이면 SpecAugment) -> backbone -> 로짓 [B, 9].

[신설 2026-09-05]

frontend 는 모델마다 다르다(조건표 3절: 각 모델의 사전학습 규격을 따른다 — 샘플레이트·mel 수·
정규화). 그 사이에 끼는 SpecAugment 만 공통이다(augment.py). 모델별 구현은 각 폴더의
model_def.py 에 있다.
"""
import torch
import torch.nn as nn

from .augment import spec_augment


class BaselineBase(nn.Module):
    #: 한 줄 설명 (로그·체크포인트 메타에 들어간다)
    MODEL_TAG = "baseline"
    DESCRIPTION = ""

    def frontend(self, wave16k):
        """[B, T] 16 kHz 파형 -> mel [B, F, T']. 리샘플·STFT·mel·정규화 전부 여기서."""
        raise NotImplementedError

    def backbone(self, mel):
        """mel [B, F, T'] -> 로짓 [B, num_classes]."""
        raise NotImplementedError

    def forward(self, wave16k, augment=False):
        mel = self.frontend(wave16k)
        if augment and self.training:
            mel = spec_augment(mel)
        return self.backbone(mel)

    def count_parameters(self, trainable_only=False):
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))
