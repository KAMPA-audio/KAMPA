#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""비교모델 효율 측정 — 파라미터 수 · 3.0초 클립 1개의 FLOPs(thop, MACs×2) · CPU 추론 지연(ms/클립).

[신설 2026-09-17 / 체크리스트 9절 "효율 표: 같은 기기에서 네 모델의 FLOPs 와 CPU 추론 시간을 3초 클립 기준으로"]

Mark5.2/model/profile_efficiency.py 와 같은 키 이름으로 JSON 을 남겨 Mark5_EXP2/model/make_paper_tables.py 의
LATENCY_ROWS(total_parameters · model_size_mb · gflops_per_clip_approx · cpu_latency_ms_per_clip)가 그대로 읽는다.
  - 입력은 16 kHz 3.0초 파형 [1, 48000] 하나(비교모델은 클립을 통째로 넣는다). 프론트엔드(리샘플·STFT·mel)가
    모델 안에 있으므로 FLOPs·지연에 그 몫이 포함된다 — Mark5.2 의 mn10 인코더 측정과 같은 기준.
  - Mark5.2 는 1.0초 세그먼트 5개(겹침)로 3초 클립을 덮으므로 "클립당" 값은 세그먼트 5개 합이다. 표에서 비교하는
    단위가 둘 다 "3초 클립 1개" 라는 뜻이고, 그 안에서 처리하는 오디오 길이는 다르다(Mark5.2 5초 분량 / 비교모델 3초).
  - 가중치는 안 실어도 FLOPs·지연은 같다(구조가 정한다). 파라미터 수는 9 클래스 헤드 기준 실측.
  - [2026-10-02] 모델이 UNUSED_MODULES(출력이 예측에 닿지 않는 모듈의 점 경로)를 밝히면 그 파라미터 수와 thop MACs 를
    total_parameters · model_size_mb · FLOPs 에서 뺀다(예: CLAP HTSAT 의 AudioSet head 는 호출되지 않고, tscam_conv 는 계산되지만
    embedding 경로 밖이다). CPU 지연은 실제 forward 를 재므로 그 연산이 들어 있다 — description 끝에 뺀 수와 함께 적는다.
    뺀 뒤의 값과 함께 전체 값(total_parameters_all_modules · gflops_per_clip_all_modules)도 남긴다. 밝히지 않은 모델은 예전과 같다.
  - CPU 고정, warmup 10 + 측정 100 (Mark5.2 와 같음). 결과의 CPU 종류는 같은 기기에서 네 모델을 잰 뒤 표에 적는다.

사용:  python profile_efficiency.py            -> plots/efficiency_profile_<model>.json
       python profile_efficiency.py --runs 30  (빠른 확인)
"""
import argparse
import json
import os
import platform
import sys
import time

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from model_def import MODEL_TAG, DISPLAY_NAME, build_model     # noqa: E402
from common.paths import CLASS_NAMES                          # noqa: E402
from common.data import SAMPLE_RATE, CLIP_SAMPLES             # noqa: E402

PLOT_DIR = os.path.join(ROOT, "plots")


def _module_at(model, path):
    """점 경로('audio_branch.head')의 하위 모듈."""
    m = model
    for part in path.split("."):
        m = getattr(m, part)
    return m


def _layer_macs(layer_info, path):
    """thop profile(..., ret_layer_info=True) 의 중첩 dict {이름: (MACs, params, 하위 dict)} 에서 점 경로 모듈의 MACs."""
    node = (None, None, layer_info)
    for part in path.split("."):
        node = node[2].get(part)
        if node is None:
            raise KeyError(f"thop 층 정보에 {path} 가 없습니다")
    return float(node[0])


def main():
    ap = argparse.ArgumentParser(description=f"{DISPLAY_NAME} 효율 측정(CPU)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device("cpu")
    model = build_model(num_classes=len(CLASS_NAMES), load_pretrained=False).to(device).eval()
    unused_modules = tuple(getattr(model, "UNUSED_MODULES", ()))
    all_params = sum(p.numel() for p in model.parameters())
    unused_params = sum(p.numel() for name in unused_modules for p in _module_at(model, name).parameters())
    total_params = all_params - unused_params
    model_size_mb = total_params * 4 / (1024 ** 2)
    dummy = torch.randn(1, CLIP_SAMPLES, device=device)

    # FLOPs — thop 은 프로파일 대상 모듈에 훅·속성을 남기므로 지연 측정용과 다른 인스턴스를 쓴다.
    try:
        from thop import profile as thop_profile
        flops_model = build_model(num_classes=len(CLASS_NAMES), load_pretrained=False).to(device).eval()
        macs_all, _, layer_info = thop_profile(flops_model, inputs=(dummy,), verbose=False, ret_layer_info=True)
        unused_macs = sum(_layer_macs(layer_info, name) for name in unused_modules)
        gmacs_all = macs_all / 1e9
        gmacs = (macs_all - unused_macs) / 1e9
        gflops = gmacs * 2
        del flops_model
    except Exception as e:                                        # thop 이 없거나 모듈을 못 세면 값을 비운다(조용히 0 을 적지 않음)
        print(f"[WARN] FLOPs 측정 실패: {e}")
        gmacs = gflops = gmacs_all = unused_macs = float("nan")

    with torch.no_grad():
        for _ in range(args.warmup):
            model(dummy)
        times = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    mean_ms = sum(times) / len(times)
    median_ms = times[len(times) // 2]

    description = getattr(model, "DESCRIPTION", "")              # [2026-09-17] 논문 효율 표 비고란(CLAP 은 전체 158M 주석 포함)
    note = "3초 클립 1개(16 kHz 파형) 기준. FLOPs≈2×MACs(thop). Mark5.2 는 1초 세그먼트 5개 합이 클립당 값."
    if unused_modules:
        _um = f"{unused_macs / 1e9:.3f} GMACs" if unused_macs == unused_macs else "MACs n/a"
        description += (f"; params/FLOPs exclude modules whose output never reaches the prediction "
                        f"({', '.join(unused_modules)}: {unused_params:,} params, {_um}); CPU latency includes that compute")
        note += f" 출력에 닿지 않는 모듈 {', '.join(unused_modules)} 의 파라미터 {unused_params:,}개·{_um} 는 뺐다(지연에는 포함)."
    result = {
        "model": MODEL_TAG,
        "display_name": DISPLAY_NAME,
        "description": description,
        "input": f"waveform 1x{CLIP_SAMPLES} ({SAMPLE_RATE // 1000} kHz, {CLIP_SAMPLES / SAMPLE_RATE:.1f}s clip, frontend included)",
        "total_parameters": int(total_params),
        "model_size_mb": round(model_size_mb, 4),
        "gmacs_per_clip": round(gmacs, 6) if gmacs == gmacs else None,
        "gflops_per_clip_approx": round(gflops, 6) if gflops == gflops else None,
        "unused_modules": list(unused_modules),                  # [2026-10-02] 위 세 값에서 뺀 모듈(없으면 빈 목록)
        "unused_parameters": int(unused_params),
        "unused_gmacs_per_clip": round(unused_macs / 1e9, 6) if unused_macs == unused_macs else None,
        "total_parameters_all_modules": int(all_params),
        "gflops_per_clip_all_modules": round(gmacs_all * 2, 6) if gmacs_all == gmacs_all else None,
        "cpu_latency_ms_per_clip": round(mean_ms, 4),
        "cpu_latency_ms_per_clip_median": round(median_ms, 4),
        "num_warmup": args.warmup,
        "num_runs": args.runs,
        "torch": torch.__version__,
        "cpu": platform.processor() or platform.machine(),
        "threads": torch.get_num_threads(),
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": note,
    }
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = args.out or os.path.join(PLOT_DIR, f"efficiency_profile_{MODEL_TAG}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[INFO] 저장: {out}")


if __name__ == "__main__":
    main()
