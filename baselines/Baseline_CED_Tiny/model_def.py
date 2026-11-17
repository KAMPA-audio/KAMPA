# -*- coding: utf-8 -*-
"""CED-Tiny (Dinkel et al., ICASSP 2024) — AudioSet 앙상블 증류로 사전학습된 오디오 트랜스포머를 9-class 전 층 파인튜닝.

[신설 2026-09-05] 조건표 3절.

    파라미터 5.5M(모델 카드, 527 헤드 포함; 9 헤드 실측 5,403,209) · AS-2M mAP 48.1 · 입력 16 kHz · 64 mel · n_fft 512
    · win 512 · hop 160 · 16x16 패치(stride 16, padding 없음) · 가변 길이. 3초 클립은 301 프레임인데 패치 conv 가 시간축
    18 패치 = 288 프레임만 쓰고 마지막 13 프레임(4.3%)은 보지 않는다 — CED 원본 동작이며 논문·조건표에 적는다.
    가중치·코드  Hugging Face mispeech/ced-tiny (Apache-2.0), revision ace276d2 (2026-09-05 기준)
                 weights/ced-tiny/ 에 그 스냅샷 사본이 있으면 그것을 쓰고, 없으면 허브에서 받는다.
                 GitHub 저장소(RicherMans/CED)의 코드는 GPL-3.0 이라 가져오지 않는다(조건표 7절 ⑬).

원격 코드(modeling_ced.py)의 CedForAudioClassification 은 파형이 아니라 **mel [B, 64, T]** 을 받는다
(main_input_name = input_values). 특징 추출기(feature_extraction_ced.py)는 torchaudio MelSpectrogram
(16 kHz, n_fft 512, win 512, hop 160, 64 mel, f_min 0, f_max None) 뒤에 AmplitudeToDB(top_db=120) 를
거는 것이 전부라, 같은 계산을 이 모듈 안에서 GPU 로 한다. 덕분에 mel 단계에 공통 SpecAugment 를
그대로 끼울 수 있다(조건표 9절 미결 5 해소).
  - AmplitudeToDB(top_db=120) 는 3차원 입력이면 배치 전체의 최댓값 기준으로 바닥을 자른다. 클립마다
    따로 자르게 [B, 1, 64, T] 로 넣어 계산한다(특징 추출기를 클립 하나씩 부른 것과 같은 값). 실제 클립의
    dB 범위는 60~66 dB 라 바닥에 닿지 않아 두 경로가 같은 값이지만(code-reviewer 실측 6개), 안전을 위해 둔다.
  - mel 규격(n_mels·hop·n_fft·win·f_min·f_max·center)은 로드한 config 에서 읽는다 — 다른 스냅샷(ced-mini 등)이
    들어오면 조용히 어긋나지 않고 여기서 멈춘다.
헤드: config.outputdim=9 로 만들고 ignore_mismatched_sizes 로 527 헤드를 버린다.
      pooling="logit" 으로 두어야 시그모이드가 아니라 원 로짓이 나온다(softmax cross-entropy 용).
      id2label/label2id 도 우리 9 클래스로 덮어써 AudioSet 이름이 남지 않게 한다.
[2026-09-05 저녁, code-reviewer 지적 반영] from_config 경로에도 code_revision 을 고정, 설정 검증을 assert 대신 예외로.
"""
import os
import sys

import torch
import torchaudio

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.base_model import BaselineBase     # noqa: E402
from common.paths import CLASS_NAMES           # noqa: E402

from model_info import MODEL_TAG, DISPLAY_NAME   # noqa: E402  (실행기가 torch 없이 읽는 가벼운 상수)
HF_ID = "mispeech/ced-tiny"
HF_REVISION = "ace276d29dd0bb3f3517b0fa8cf300738c409019"
LOCAL_SNAPSHOT = os.path.join(ROOT, "weights", "ced-tiny")

SAMPLE_RATE = 16000
TOP_DB = 120
# 사전학습 규격(모델 카드·preprocessor_config.json). 로드한 config 와 대조하는 기대값이다.
EXPECTED = {"n_mels": 64, "hop_size": 160, "n_fft": 512, "win_size": 512, "f_min": 0, "f_max": 8000, "center": True,
            "patch_size": 16, "patch_stride": 16, "target_length": 1012}


def _source():
    if os.path.isdir(LOCAL_SNAPSHOT):
        need = ("config.json", "model.safetensors", "modeling_ced.py", "configuration_ced.py")
        missing = [f for f in need if not os.path.isfile(os.path.join(LOCAL_SNAPSHOT, f))]
        if missing:
            raise FileNotFoundError(f"[ERROR] weights/ced-tiny 스냅샷에 파일이 빠져 있습니다: {missing}")
        return LOCAL_SNAPSHOT
    return HF_ID


class CEDTinyBaseline(BaselineBase):
    MODEL_TAG = MODEL_TAG
    DESCRIPTION = "CED-Tiny (audio transformer, 16x16 patches), AudioSet pretrained, outputlayer(192->9), full fine-tuning"

    def __init__(self, num_classes=9, load_pretrained=True, source=None):
        super().__init__()
        from transformers import AutoConfig, AutoModelForAudioClassification
        src = source or _source()
        id2label = {i: c for i, c in enumerate(CLASS_NAMES[:num_classes])}
        kw = dict(trust_remote_code=True, outputdim=num_classes, pooling="logit",
                  id2label=id2label, label2id={c: i for i, c in id2label.items()})
        if src == HF_ID:
            kw["revision"] = HF_REVISION
        self.source = src
        self.pretrained_loaded = False
        if load_pretrained:
            self.core = AutoModelForAudioClassification.from_pretrained(src, ignore_mismatched_sizes=True, **kw)
            self.pretrained_loaded = True
        else:
            cfg = AutoConfig.from_pretrained(src, **kw)
            fc_kw = {"trust_remote_code": True}
            if src == HF_ID:
                fc_kw["code_revision"] = HF_REVISION      # 원격 코드도 같은 revision 으로(전에는 main 을 받았다)
            self.core = AutoModelForAudioClassification.from_config(cfg, **fc_kw)
        cfg = self.core.config
        if cfg.outputdim != num_classes or cfg.pooling != "logit":
            raise ValueError(f"[ERROR] CED config 반영 실패: outputdim={cfg.outputdim}, pooling={cfg.pooling}")
        bad = {k: getattr(cfg, k, None) for k, v in EXPECTED.items() if getattr(cfg, k, None) != v}
        if bad:
            raise ValueError(f"[ERROR] CED 스냅샷의 규격이 기대값과 다릅니다(ced-tiny 가 맞는지 확인): {bad} vs {EXPECTED}")
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=cfg.n_fft, win_length=cfg.win_size, hop_length=cfg.hop_size,
            f_min=cfg.f_min, f_max=cfg.f_max, n_mels=cfg.n_mels, center=cfg.center)
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB)

    def frontend(self, wave16k):
        mel = self.melspec(wave16k)                          # [B, 64, T']
        return self.to_db(mel.unsqueeze(1)).squeeze(1)       # 클립별 top_db

    def backbone(self, mel):
        return self.core(input_values=mel).logits            # pooling="logit" -> [B, 9] 원 로짓


def build_model(num_classes=9, load_pretrained=True, source=None):
    return CEDTinyBaseline(num_classes=num_classes, load_pretrained=load_pretrained, source=source)
