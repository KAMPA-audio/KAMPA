# -*- coding: utf-8 -*-
"""PANNs CNN6 (Kong et al., IEEE/ACM TASLP 2020) — AudioSet 사전학습 가중치로 9-class 전 층 파인튜닝.

[신설 2026-09-05] 조건표 3절.

    파라미터 4.8M(논문 표기, 527 헤드 포함) · 실측 학습 대상 4,571,721(527 헤드 제거·9 헤드 추가; STFT·mel 커널 1,083,456 은 고정)
    사전학습 mAP 0.343 · 입력 32 kHz · n_fft 1024 · hop 320 · 64 mel · 50~14,000 Hz
    가중치  weights/Cnn6_mAP=0.343.pth  (Zenodo 3987831, 23.7 MB, MIT)
    코드    third_party/panns/models.py · pytorch_utils.py  (저장소 qiuqiangkong/audioset_tagging_cnn 사본, MIT)
            + pip torchlibrosa (STFT·log-mel, MIT)

원본 Cnn6.forward 를 그대로 부르지 않고 부품(spectrogram_extractor · logmel_extractor · bn0 ·
conv_block1~4 · fc1)을 같은 순서로 다시 잇는다. 이유 두 가지:
  (1) 원본은 학습 중 자기 SpecAugmentation(time 64·freq 8 각 2개)을 걸고 mixup 도 받는다.
      우리는 세 모델 공통 SpecAugment(common/augment.py) 하나만 쓰기로 했다(조건표 2절).
      원본에서 마스크는 **bn0 뒤**에 걸린다(models.py Cnn6.forward: bn0 -> spec_augmenter). 같은 순서가 되게
      bn0 까지를 frontend 에 두고, 공통 SpecAugment 는 그 뒤(backbone 직전)에 걸린다.
      [2026-09-05 저녁, code-reviewer 지적 반영 — 처음 판은 bn0 가 backbone 에 있어 마스크가 bn0 앞에 걸렸다]
  (2) 분류 헤드: 저장소 finetune_template.py 의 Transfer_Cnn14 방식대로 fc_audioset(527) 을 버리고
      임베딩(512) 뒤에 새 fc_transfer(9) 를 둔다. dropout 위치·비율(0.2/0.5)은 원본 그대로.
16 kHz 원본은 32 kHz 로 리샘플해 넣는다(사전학습 mel 필터와 맞추기 위함, 조건표 7절 ⑩). 그 결과 64 mel 중
8 kHz 위의 9개 bin(55~63)은 거의 비어 있다 — 사전학습 규격을 따르는 대가이며 논문에 한 줄 적는다.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

ROOT = os.path.dirname(os.path.abspath(__file__))
THIRD = os.path.join(ROOT, "third_party", "panns")
for p in (ROOT, THIRD):
    if p not in sys.path:
        sys.path.insert(0, p)

from models import Cnn6, init_layer          # noqa: E402  (third_party/panns/models.py, 원본 그대로)
from common.base_model import BaselineBase   # noqa: E402

from model_info import MODEL_TAG, DISPLAY_NAME   # noqa: E402  (실행기가 torch 없이 읽는 가벼운 상수)
PRETRAINED_PATH = os.path.join(ROOT, "weights", "Cnn6_mAP=0.343.pth")

SAMPLE_RATE = 32000
WINDOW_SIZE = 1024
HOP_SIZE = 320
MEL_BINS = 64
FMIN = 50
FMAX = 14000
AUDIOSET_CLASSES = 527


class PANNsCnn6Baseline(BaselineBase):
    MODEL_TAG = MODEL_TAG
    DESCRIPTION = "PANNs CNN6, AudioSet pretrained, fc_transfer(512->9), full fine-tuning"

    def __init__(self, num_classes=9, load_pretrained=True, pretrained_path=PRETRAINED_PATH):
        super().__init__()
        self.core = Cnn6(SAMPLE_RATE, WINDOW_SIZE, HOP_SIZE, MEL_BINS, FMIN, FMAX, AUDIOSET_CLASSES)
        self.source = pretrained_path
        self.pretrained_loaded = False
        if load_pretrained:
            if not os.path.isfile(pretrained_path):
                raise FileNotFoundError(f"[ERROR] PANNs 가중치가 없습니다: {pretrained_path}")
            ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)   # 옛 pickle 포맷(zip 아님)이라 False 가 맞다
            if not isinstance(ckpt, dict) or "model" not in ckpt:
                raise RuntimeError(f"[ERROR] PANNs 체크포인트 형식이 아닙니다('model' 키 없음): {pretrained_path} — "
                                   "Zenodo 3987831 의 Cnn6_mAP=0.343.pth 인지 확인하십시오.")
            self.core.load_state_dict(ckpt["model"])       # 키가 하나라도 안 맞으면 여기서 멈춘다(strict)
            self.pretrained_loaded = True
        # AudioSet 527 헤드는 쓰지 않으므로 떼어낸다(파라미터 수 270,351 이 표에 섞이지 않게).
        # load_pretrained=False 경로(eval.py)에서도 같은 구조가 되어 state_dict 키가 일치한다.
        self.core.fc_audioset = nn.Identity()
        # 원본 내장 SpecAugmentation 은 쓰지 않는다(공통 SpecAugment 로 대체). 파라미터가 없어 state_dict 에는 영향 없음.
        del self.core.spec_augmenter
        self.fc_transfer = nn.Linear(self.core.fc1.out_features, num_classes, bias=True)   # 512 (원본 fc1 에서 읽음)
        init_layer(self.fc_transfer)

    def frontend(self, wave16k):
        # 리샘플은 함수형으로 — Resample 모듈은 커널을 버퍼로 state_dict 에 넣어 torchaudio 버전이 갈리면 로드가 깨진다.
        x = torchaudio.functional.resample(wave16k, 16000, SAMPLE_RATE)   # [B, T@32k]
        x = self.core.spectrogram_extractor(x)       # [B, 1, T', n_fft/2+1]
        x = self.core.logmel_extractor(x)            # [B, 1, T', 64]  (log-mel, dB)
        x = x.transpose(1, 3)                        # [B, 64, T', 1]  bn0 는 64 mel 채널에 걸린다(원본과 같은 transpose 왕복)
        x = self.core.bn0(x)
        x = x.transpose(1, 3)                        # [B, 1, T', 64]
        return x[:, 0].transpose(1, 2)               # [B, 64, T']  (bn0 까지 끝난 값. 공통 SpecAugment 는 이 뒤에 걸린다)

    def backbone(self, mel):
        c = self.core
        x = mel.transpose(1, 2).unsqueeze(1)         # [B, 1, T', 64]  원본 레이아웃(batch, 1, time, mel)
        # 원본은 여기(bn0 뒤)에서 spec_augmenter · mixup 을 걸었다. 공통 SpecAugment 가 그 자리를 대신한다.
        x = c.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = c.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = c.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = c.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)                     # 주파수축 평균
        x1, _ = torch.max(x, dim=2)                  # 시간축 max + mean (전역 풀링 -> 길이 제약 없음)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(c.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        return self.fc_transfer(embedding)


def build_model(num_classes=9, load_pretrained=True):
    return PANNsCnn6Baseline(num_classes=num_classes, load_pretrained=load_pretrained)
