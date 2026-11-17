# -*- coding: utf-8 -*-
"""데이터 — dataset_*.csv 를 읽어 3초 파형 클립을 배치로 내준다.

[신설 2026-09-05]

Mark5 는 클립을 1초 세그먼트 5개로 잘라 세그먼트 단위로 학습·집계하지만(구조의 일부),
비교모델 셋은 3초 클립 전체를 한 번에 받는다(조건표 2절 '평가 단위'). 세 모델 모두
시간축을 전역 풀링하거나 가변 길이를 지원하므로 3초를 그대로 넣는다.

파형은 16 kHz 로 읽고 정확히 48,000 샘플(3.0초)로 맞춘다(길면 앞에서 자르고 짧으면 0 을
붙인다). 4,860개는 48,000 또는 48,001 샘플이라 실제로는 최대 1 샘플만 잘린다. 32 kHz 가 필요한 모델
(PANNs·EfficientAT)은 model_def 안에서 리샘플한다 — 데이터 쪽은 모델과 무관하게 둔다.

preload=True 면 파형 전부를 메모리에 올린다(labeled 2,000 + 합의 라벨 약 1,800 = 약 730MB).
드라이브에서 매 배치 wav 를 읽으면 I/O 가 학습을 잡아먹기 때문이다(Mark5_EXP 도 파싱
결과를 메모리에 둔다).
"""
import csv
import os

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .paths import CLASS_NAMES, LABEL_MAP, resolve_audio_path

SAMPLE_RATE = 16000
CLIP_SECONDS = 3.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)   # 48,000


def read_index(csv_path):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_clip(path):
    """wav -> float32 [CLIP_SAMPLES]. 모노화 규칙은 Mark5 utils/parser_utils.py 와 같다
    (2차원이면 작은 축을 채널로 보고 평균)."""
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        t, c = wav.shape
        wav = wav.mean(axis=1) if t >= c else wav.mean(axis=0)
    if sr != SAMPLE_RATE:
        import torchaudio.functional as AF
        wav = AF.resample(torch.from_numpy(wav), sr, SAMPLE_RATE).numpy()
    if len(wav) >= CLIP_SAMPLES:
        wav = wav[:CLIP_SAMPLES]
    else:
        wav = np.concatenate([wav, np.zeros(CLIP_SAMPLES - len(wav), dtype=np.float32)])
    return np.ascontiguousarray(wav, dtype=np.float32)


class ClipDataset(Dataset):
    """rows: [{"path": ..., "label": ..., ("confidence": ..., "source": ...)}, ...]"""

    def __init__(self, rows, data_root, name="", preload=True, require_labels=True, verbose=True, strict=True):
        """strict=True(기본): 오디오를 하나라도 못 찾으면 즉시 멈춘다. 드라이브 마운트가 부분적으로 풀리거나
        --data_root 가 틀린 채로 적은 표본의 지표가 논문 표에 들어가는 것을 막기 위함(code-reviewer 지적:
        전에는 절반 미만 누락이면 경고 한 줄만 찍고 진행했다)."""
        self.name = name
        self.items = []          # (local_path, label_idx, meta)
        missing, bad_label = 0, {}
        for r in rows:
            label = r.get("label")
            if require_labels and label not in LABEL_MAP:
                bad_label[label] = bad_label.get(label, 0) + 1
                continue
            local = resolve_audio_path(r["path"], data_root)
            if local is None:
                missing += 1
                continue
            meta = {"path": r["path"], "local_path": local, "label": label,
                    "source": r.get("source", "labeled"),
                    "confidence": float(r["confidence"]) if r.get("confidence") not in (None, "") else float("nan")}
            self.items.append((local, LABEL_MAP.get(label, -1), meta))
        if verbose:
            print(f"[DATA] {name}: {len(self.items)}개 사용"
                  + (f", 파일 없음 {missing}개" if missing else "")
                  + (f", 9-class 밖 라벨 {bad_label}" if bad_label else ""))
        if missing and (strict or missing * 2 >= len(rows)):      # '절반 이상' — 정수 나눗셈(3//2=1)이면 3행 중 1행도 걸려서 이렇게 쓴다
            raise FileNotFoundError(
                f"[ERROR] {name}: 오디오 {missing}/{len(rows)}개를 못 찾았습니다. "
                f"--data_root 가 Mark5_EXP 폴더(코랩: 드라이브 마운트)를 가리키는지 확인하십시오 (지금: {data_root}). "
                + ("" if strict else "(strict=False 라 절반 이상일 때만 멈춥니다)"))
        if require_labels and len(self.items) == 0:
            raise ValueError(f"[ERROR] {name}: 유효한 클립이 0개입니다.")
        self.waves = None
        if preload:
            arr = np.empty((len(self.items), CLIP_SAMPLES), dtype=np.float32)
            for i, (p, _, _) in enumerate(self.items):
                arr[i] = load_clip(p)
                if verbose and (i + 1) % 500 == 0:
                    print(f"[DATA] {name}: {i + 1}/{len(self.items)} 로드")
            self.waves = torch.from_numpy(arr)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, meta = self.items[idx]
        wave = self.waves[idx] if self.waves is not None else torch.from_numpy(load_clip(path))
        return wave, label, meta

    def label_counts(self):
        counts = {c: 0 for c in CLASS_NAMES}
        for _, l, _ in self.items:
            if l >= 0:
                counts[CLASS_NAMES[l]] += 1
        return counts


def collate(batch):
    waves, labels, metas = zip(*batch)
    return torch.stack(waves, dim=0), torch.tensor(labels, dtype=torch.long), list(metas)


def select_labeled_subset(rows, labeled_n, seed, classes):
    """[신설 2026-09-17 / 9절] labeled 서브샘플 — Mark5_EXP2/model/train_mark5.py select_labeled_subset 과 **같은 규칙**
    (클래스별 n//C 개 + 나머지는 앞 클래스부터 1개씩, random.Random(seed) 로 클래스별 순서를 섞어 앞에서 자름).
    dataset_labeled.csv 가 Mark5.2 의 것과 같은 파일이면 같은 seed·n 에 같은 클립이 뽑혀 seed 짝지은 비교가 된다.
    반환: (뽑힌 행, 남은 행)."""
    import random
    n = int(labeled_n)
    classes = list(classes)
    if n <= 0 or n > len(rows):
        raise SystemExit(f"[ERROR] labeled_n={n} 은 1 이상, 실제 labeled 개수({len(rows)}) 이하여야 합니다.")
    by_cls = {c: [r for r in rows if r.get("label") == c] for c in classes}
    rng = random.Random(int(seed))
    for c in classes:
        rng.shuffle(by_cls[c])
    base, extra = divmod(n, len(classes))
    picked, rest = [], []
    for i, c in enumerate(classes):
        k = base + (1 if i < extra else 0)
        if k > len(by_cls[c]):
            raise SystemExit(f"[ERROR] labeled_n={n}: 클래스 {c} 에 {k}개가 필요한데 {len(by_cls[c])}개뿐입니다.")
        picked += by_cls[c][:k]
        rest += by_cls[c][k:]
    return picked, rest


def subsample_nested(rows, n, seed):
    """[신설 2026-09-17 / 9절] seed 고정으로 섞어 앞에서 n 개(누적 구조). Mark5_EXP2 의 unlabeled 서브샘플과 같은 규칙
    (random.Random(seed) 로 0..len-1 을 섞어 앞 n 개, 원래 순서로 정렬). 같은 순서의 목록을 주면 같은 원소가 뽑힌다."""
    import random
    n = int(n)
    if n < 0 or n > len(rows):
        raise SystemExit(f"[ERROR] unlabeled_n={n} 은 0 이상, 실제 개수({len(rows)}) 이하여야 합니다.")
    rng = random.Random(int(seed))
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    return [rows[i] for i in sorted(idx[:n])]


def clip_key(path):
    """클립을 가리키는 열쇠 = 파일 이름. 인덱스 CSV 마다 앞 경로(코랩 드라이브·로컬)가 달라도 같은 클립이면 같은 값."""
    return os.path.basename(str(path).replace("\\", "/"))


def restrict_to_unlabeled_subset(consensus_rows, unlabeled_rows, n, seed):
    """[신설 2026-10-01] 데이터 효율 곡선(--unlabeled_n)용. 라벨 없는 클립 목록(dataset_unlabeled.csv 의 행 순서 그대로)에서
    subsample_nested 로 n 개를 뽑고, 합의 라벨 가운데 그 n 개에 속한 것만 돌려준다.

    Mark5_EXP2/model/train_mark5.py 는 라벨 없는 클립 n 개를 먼저 뽑고 그 안에서 전문가 합의로 채택된 것만 가짜 정답으로 쓴다
    (2026-10-01 로그: 500 개 중 351~383 개, 1,000 개 중 722~737 개 — 전원 거절 others 포함). 전에는 여기서 이미 채택된 합의 라벨
    1,338 개 중 n 개를 뽑아, 같은 _u500 인데 비교모델이 가짜 정답을 약 1.35 배 더 받았다. 같은 dataset_unlabeled.csv·같은 seed 면
    이 함수가 고르는 클립은 Mark5.2 가 가짜 정답을 받는 클립과 같다(2026-10-01 EXP2 학습 로그 15회와 개수 일치 확인).
    합의 라벨에 라벨 없는 목록 밖의 클립이 있으면 재료가 뒤섞인 것이라 멈춘다."""
    all_keys = {clip_key(r["path"]) for r in unlabeled_rows}
    stray = [r["path"] for r in consensus_rows if clip_key(r["path"]) not in all_keys]
    if stray:
        raise SystemExit(f"[ERROR] 합의 라벨 {len(stray)}개가 dataset_unlabeled.csv 에 없는 클립입니다(예: {stray[0]}). "
                         "합의 라벨과 dataset_unlabeled.csv 가 같은 데이터 판인지 확인하십시오.")
    picked = {clip_key(r["path"]) for r in subsample_nested(unlabeled_rows, n, seed)}
    return [r for r in consensus_rows if clip_key(r["path"]) in picked]


def load_consensus_rows(csv_path, min_conf, source="consensus", what="합의 라벨"):
    """make_consensus_labels.py(Mark5_EXP) 가 만든 CSV: path,label,confidence[,n_segments,...,generated_at].
    CSV 는 이미 선별된 것이지만, 임계값을 한 번 더 걸어 두어 다른 임계값으로 만든 파일이
    섞여 들어와도 조건이 어긋나지 않게 한다.

    [2026-09-05 저녁, code-reviewer 지적] 열이 빠졌거나 값이 비면 전부 탈락해 --cond pl 이 조용히 지도학습이
    되던 경로를 막는다: 필수 열이 없으면 멈추고, 빈 확신도는 세어서 알리고, 남는 행이 0 이면 멈춘다.
    generated_at 열이 있으면 찍어 두어 어느 생성본으로 학습했는지 로그에 남긴다.

    [2026-09-06] 천장(--cond ceil)용 ceiling_labels.csv(make_ceiling_labels.py, confidence 전부 1.0)도
    같은 스키마라 이 함수로 읽는다. source 로 행의 출처 표시("consensus" / "ceiling")를, what 으로 메시지의
    이름을 바꾼다.

    [2026-10-01] 합의 라벨 CSV 에 origin 열이 생겼다(make_consensus_labels.py --others_from_reject): "consensus" = 전문가 합의로 채택,
    "reject_others" = 전문가 8명이 모두 거절해 others 로 채택(Mark5.2 학습의 OR 과 같은 규칙, confidence = 1 − 최대 확신도라 0.8 이상).
    두 종류 모두 가중치 없이 hard label 로 쓴다(Mark5.2 는 확신도 가중을 쓰지만 비교모델은 처음부터 가중치 없이 써 왔다).
    origin 열이 없는 옛 CSV 는 전부 "consensus" 로 본다."""
    rows = read_index(csv_path)
    if not rows:
        raise ValueError(f"[ERROR] {what} CSV 가 비어 있습니다: {csv_path}")
    missing_cols = [c for c in ("path", "label", "confidence") if c not in rows[0]]
    if missing_cols:
        raise ValueError(f"[ERROR] {what} CSV 에 필수 열이 없습니다 {missing_cols}: {csv_path} "
                         "(Mark5_EXP model/make_consensus_labels.py 또는 make_ceiling_labels.py 가 만든 파일인지 확인)")
    kept, n_blank, n_bad_label, n_below = [], 0, 0, 0
    for r in rows:
        raw = (r.get("confidence") or "").strip()
        if raw == "":
            n_blank += 1
            continue
        conf = float(raw)
        if not (conf >= min_conf):
            n_below += 1
            continue
        if r.get("label") not in LABEL_MAP:
            n_bad_label += 1
            continue
        kept.append({"path": r["path"], "label": r["label"], "confidence": raw, "source": source,
                     "generated_at": r.get("generated_at", ""),     # [2026-09-06 리뷰 반영] train_meta 의 consensus_generated_at 이 늘 "" 이던 것
                     "origin": (r.get("origin") or "consensus")})  # [2026-10-01] consensus / reject_others(전원 거절 → others)
    stamp = rows[0].get("generated_at", "")
    n_rej = sum(1 for r in kept if r["origin"] == "reject_others")
    print(f"[DATA] {what} {csv_path}: {len(rows)}행 중 확신도 >= {min_conf} 인 {len(kept)}행 사용"
          + (f" (합의 {len(kept) - n_rej} + 전원 거절→others {n_rej})" if n_rej else "")
          + (f", 확신도 빈칸 {n_blank}" if n_blank else "") + (f", 임계값 미달 {n_below}" if n_below else "")
          + (f", 9-class 밖 라벨 {n_bad_label}" if n_bad_label else "") + (f" | generated_at={stamp}" if stamp else ""))
    if not kept:
        raise ValueError(f"[ERROR] {what} CSV 에서 쓸 수 있는 행이 0개입니다: {csv_path} — 이 조건을 돌릴 수 없습니다.")
    return kept
