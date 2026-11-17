# -*- coding: utf-8 -*-
"""EfficientAT mn10 (Schmid et al., ICASSP 2023) — AudioSet 사전학습 MobileNetV3(폭 1.0)를 9-class 전 층 파인튜닝.

[신설 2026-09-05] 조건표 3절.

    파라미터 4.88M(논문 표기, 527 헤드 포함; 9 헤드로 바꾼 실측 4,213,273) · 0.54 GMACs/10초 · 사전학습 mAP 47.1
    입력 32 kHz · 128 mel · win 800 · hop 320 · n_fft 1024. 16 kHz 원본을 32 kHz 로 올려 넣으므로 128 mel 중 8 kHz
    위의 24 bin 은 거의 비어 있다(code-reviewer 실측) — 사전학습 규격을 따르는 대가이며 논문에 한 줄 적는다.
    가중치  weights/mn10_as_mAP_471.pt  (저장소 릴리스 v0.0.1, 19.7 MB, MIT)
    코드    third_party/efficientat/models/mn/*.py · models/preprocess.py · helpers/utils.py
            (저장소 fschmid56/EfficientAT 사본, MIT; 원본 import 경로를 살리려고 폴더 구조를 그대로 둠)

전처리는 원본 AugmentMelSTFT(models/preprocess.py)와 같은 계산을 이 파일 안에 다시 적었다.
  - 저자 기본값: preemphasis(-0.97) -> STFT(1024, hop 320, win 800, hann) -> power -> Kaldi mel 128
    -> log(x+1e-5) -> (x+4.5)/5 정규화. fmax 는 저자 코드에서 None 이면 sr/2 - fmax_aug_range/2 = 15,000 Hz
    (평가 시 값)이라 15,000 으로 고정했다.
  - 다시 적은 이유 (1) 원본은 학습 중 freqm/timem 마스크와 fmin/fmax 무작위 이동을 건다 —
    공통 SpecAugment 만 쓰기로 했으므로 전부 끈다. (2) 원본의 torch.stft(return_complex=False) 는
    최신 torch 에서 막힐 수 있어 return_complex=True 로 받아 |X|^2 을 계산한다(값은 같다).
분류 헤드는 저자 방식대로 get_model(num_classes=9, head_type="mlp") 로 마지막 Linear 만 새로 만들고,
사전학습 state_dict 에서 classifier.5.* 두 키만 빼고 나머지를 전부 strict 하게 맞춘다.
"""
import contextlib
import io
import os
import sys

import torch
import torch.nn.functional as F
import torchaudio

ROOT = os.path.dirname(os.path.abspath(__file__))
THIRD = os.path.join(ROOT, "third_party", "efficientat")
for p in (ROOT, THIRD):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.mn.model import get_model          # noqa: E402  (third_party/efficientat, 원본 그대로)
from common.base_model import BaselineBase     # noqa: E402

from model_info import MODEL_TAG, DISPLAY_NAME   # noqa: E402  (실행기가 torch 없이 읽는 가벼운 상수)
PRETRAINED_PATH = os.path.join(ROOT, "weights", "mn10_as_mAP_471.pt")

SAMPLE_RATE = 32000
N_MELS = 128
WIN_LENGTH = 800
HOP_SIZE = 320
N_FFT = 1024
FMIN = 0.0
FMAX = 15000.0     # 저자 코드: fmax=None -> sr//2 - fmax_aug_range//2 = 16000 - 1000
HEAD_KEYS = ("classifier.5.weight", "classifier.5.bias")


class EfficientATmn10Baseline(BaselineBase):
    MODEL_TAG = MODEL_TAG
    DESCRIPTION = "EfficientAT mn10 (MobileNetV3 w1.0), AudioSet pretrained, mlp head(->9), full fine-tuning"

    def __init__(self, num_classes=9, load_pretrained=True, pretrained_path=PRETRAINED_PATH):
        super().__init__()
        with contextlib.redirect_stdout(io.StringIO()):      # get_model 이 모델 전체를 print 한다
            self.core = get_model(num_classes=num_classes, pretrained_name=None, width_mult=1.0,
                                  head_type="mlp", se_dims="c")
        self.source = pretrained_path
        self.pretrained_loaded = False
        if load_pretrained:
            if not os.path.isfile(pretrained_path):
                raise FileNotFoundError(f"[ERROR] EfficientAT 가중치가 없습니다: {pretrained_path}")
            sd = torch.load(pretrained_path, map_location="cpu", weights_only=True)   # 텐서만 든 파일(실측)이라 True 로 충분
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            if HEAD_KEYS[1] not in sd:
                raise RuntimeError(f"[ERROR] mlp 헤드(classifier.5.*)가 든 EfficientAT 가중치가 아닙니다: {pretrained_path}")
            n_pre = int(sd[HEAD_KEYS[1]].numel())
            popped = n_pre != num_classes
            if popped:
                for k in HEAD_KEYS:
                    sd.pop(k, None)
            res = self.core.load_state_dict(sd, strict=False)     # 이름이 같고 shape 이 다르면 여기서 RuntimeError 가 난다(실측)
            unexpected = list(res.unexpected_keys)
            missing = [k for k in res.missing_keys if not (popped and k in HEAD_KEYS)]
            if unexpected or missing:
                raise RuntimeError(f"[ERROR] EfficientAT 가중치 키 불일치 — missing={missing}, unexpected={unexpected}")
            self.pretrained_loaded = True
        self.register_buffer("window", torch.hann_window(WIN_LENGTH, periodic=False), persistent=False)
        self.register_buffer("preemphasis", torch.as_tensor([[[-0.97, 1.0]]]), persistent=False)
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            N_MELS, N_FFT, SAMPLE_RATE, FMIN, FMAX, vtln_low=100.0, vtln_high=-500.0, vtln_warp_factor=1.0)
        self.register_buffer("mel_basis", F.pad(mel_basis, (0, 1), mode="constant", value=0), persistent=False)

    def frontend(self, wave16k):
        # 리샘플은 함수형으로 — Resample 모듈은 커널을 버퍼로 state_dict 에 넣어 torchaudio 버전이 갈리면 로드가 깨진다.
        x = torchaudio.functional.resample(wave16k, 16000, SAMPLE_RATE)         # [B, T@32k]
        x = F.conv1d(x.unsqueeze(1), self.preemphasis).squeeze(1)
        spec = torch.stft(x, N_FFT, hop_length=HOP_SIZE, win_length=WIN_LENGTH, center=True,
                          normalized=False, window=self.window, return_complex=True)
        power = spec.real ** 2 + spec.imag ** 2                                 # [B, 513, T']
        mel = torch.matmul(self.mel_basis, power)                               # [B, 128, T']
        mel = (mel + 0.00001).log()
        return (mel + 4.5) / 5.0

    def backbone(self, mel):
        logits, _features = self.core(mel.unsqueeze(1))                         # [B, 9], [B, C]
        return logits


def build_model(num_classes=9, load_pretrained=True):
    return EfficientATmn10Baseline(num_classes=num_classes, load_pretrained=load_pretrained)
