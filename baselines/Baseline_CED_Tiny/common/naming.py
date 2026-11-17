# -*- coding: utf-8 -*-
"""run_tag 와 산출물 경로 규칙 — 한 곳에만 둔다.

[신설 2026-09-05 저녁, code-reviewer 지적] train.py 와 run_experiments.py 가 같은 규칙을 따로 갖고 있었다.
한쪽만 고치면 산출물이 미아가 된다. torch 를 물지 않는 모듈이라 실행기도 그대로 import 한다.

    run_tag = <model>_<cond>_adam[_l<n>][_u<n>]_s<seed>[_smoke]     ([2026-10-25~] 지금 절차, 예: ced_tiny_sup_adam_l200_s1)
            = <model>_<cond>_lr<lr>[_l<n>][_u<n>]_s<seed>[_smoke]   (2026-10-21까지. 아래 예시는 이 옛 형식)
    예: ced_tiny_sup_lr5e-05_s1        본실험(labeled 전량)
        ced_tiny_pl_lr5e-05_l200_s1    [2026-09-17 / 9절] labeled 200 만 쓴 회차(라벨 효율 표). Mark5_EXP2 의 _l<n> 과 같은 뜻
        ced_tiny_pl_lr5e-05_l200_u500_s1  labeled 200 + 라벨 없는 클립 500 개 안의 합의 라벨(데이터 효율 곡선). Mark5_EXP2 의 _u<n> 과 같은 뜻
                                          [2026-10-01] 전에는 "합의 라벨 500 개" 였다 — 아래 UNLABELED_SUBSAMPLE_RULE 참조
        ced_tiny_sup_lr0.0001_s1_smoke  동작 확인용(--max_clips). 본실험 태그와 절대 겹치지 않는다.
        efficientat_mn10_fixmatch_lr0.0001_l200_s1  [2026-10-07] 표준 반지도 기준선(cond = 방법 이름, 아래 SSL_METHODS)

[변경 2026-10-25 / A안] 10-22 에 바꾼 Prodigy(자동 학습률, 태그 _pd)를 걷어내고 Adam 으로 되돌렸다. Prodigy 의 보폭이 Adam 학습률 1e-4 의
13~780배로 커져 사전학습 가중치가 지워졌다(정답 2,000 체크포인트에서 절댓값 0.001 미만 가중치 CED-Tiny 96.8%·EfficientAT 60.7%,
test Acc Adam 대비 −29.54·−9.30%p). 학습률은 새로 고르지 않고 Adam 시절 선정 기록(plots/lr_selected*.json)이 고른 값을 고정한다(adam_lr).
학습률 자리(lr)에 None 을 주면 지금 절차 회차이고 태그의 _lr<lr> 자리에 _adam 이 붙는다(예: ced_tiny_sup_adam_l200_s1). 옛 Adam 태그
(_lr0.0001)와 이름을 나눈 것은, 10-25 아이 말소리 교체가 데이터 목록을 그대로 두고 오디오만 바꿔 인덱스 지문으로는 옛 결과를 가려낼 수
없기 때문이다. 옛 Adam 태그(_lr<lr>)는 계속 읽고, Prodigy 태그(_pd)는 읽지 않는다(parse_tag 가 None — 집계에서 빠진다).
"""
import re
import os

# [신설 2026-10-01] --unlabeled_n 이 무엇을 n 개로 줄이는지의 판 이름. train.py 가 train_meta 에 적고 run_experiments.py 가 재개 판정 때 대조한다.
#   옛 동작(2026-09-17~10-01, 판 이름 없음): 이미 채택된 **합의 라벨**에서 n 개를 뽑았다.
#   지금: **라벨 없는 클립**(dataset_unlabeled.csv)에서 n 개를 뽑고 그 안에 있는 합의 라벨만 쓴다 — Mark5_EXP2/model/train_mark5.py 의
#        unlabeled 서브샘플과 같은 규칙·같은 순서라, 같은 seed 면 Mark5.2 가 가짜 정답을 받는 클립과 같은 클립이다.
#   run_tag(_u<n>)는 그대로라, 이 값이 없거나 다른 옛 산출물은 완료로 치지 않는다(다시 돈다).
UNLABELED_SUBSAMPLE_RULE = "unlabeled_clips_v20261001"

# [신설 2026-10-07 / 논문 표 1] 표준 반지도 기준선 4종 — 각 방법이 곧 학습 조건(cond)이다(common/ssl.py · common/ssl_loop.py).
#   예: efficientat_mn10_fixmatch_lr0.0001_l200_s1. sup·pl·ceil 과 같은 run_tag 규칙을 쓴다.
#   ssl_impl_version(방법): train.py 가 train_meta 에 적고 run_experiments.py(재개 판정)·make_ssl_table.py(표)가 대조한다 — 구현을
#   고치면 그 방법의 판을 바꿔 옛 구현으로 만든 산출물이 완료로 남거나 표에 섞이지 않게 한다. torch 를 물지 않는 모듈이라
#   실행기·집계도 그대로 import 한다.
#   [2026-10-08] 판을 방법별로 나눴다. 고친 방법만 다시 돌고, 바뀐 것이 없는 방법(FixMatch)은 10-07 결과를 그대로 쓴다.
#     pseudolabel·meanteacher: 섞은 배치(정답 + 라벨 없음, 배치 16)에서 정답 클립만 따로 순전파하던 것을 배치 전체 한 번 순전파로
#       (정답 라벨 200 이면 정답이 배치당 약 1.5개라 BN 이 1~2개 클립으로 계산됐다 — 10-07 결과에서 두 방법만 하한보다 낮았던 원인).
#     meanteacher·freematch: EMA 감쇠 0.999 -> 0.99 (USB 0.999 는 102,400 걸음 기준. 우리는 조기 종료까지 수천 걸음이라 Mark5.3 FM 과
#       같은 규칙으로 맞췄다 — Mark5.3/vild/vild_config.py m53_fm_ema·m53_fm_momentum).
SSL_METHODS = ("pseudolabel", "meanteacher", "fixmatch", "freematch")
SSL_IMPL_VERSIONS = {"pseudolabel": "ssl_v20261008", "meanteacher": "ssl_v20261008",
                     "fixmatch": "ssl_v20261007", "freematch": "ssl_v20261008"}


def ssl_impl_version(method):
    """그 표준 반지도 방법의 지금 구현 판(train_meta·학습률 기록·체크포인트에 적고 대조하는 값)."""
    return SSL_IMPL_VERSIONS[method]


# [변경 2026-10-25 / A안] 지금 학습 코드가 쓰는 옵티마이저. train.py 가 train_meta 에 적고 run_experiments.py 가 재개 판정 때 대조한다.
OPTIMIZER_NAME = "adam"
# lr 자리에 넣는 '지금 절차' 표시(태그 _adam). None 도 같은 뜻이지만 표·곡선 스크립트(make_ssl_table.py 등)는 None 을 '결과 없음'으로 쓰므로 이 값을 넘긴다.
CUR = "adam"
# [신설 2026-10-25 / A안] 학습률(고정). Adam 시절 선정 기록 그대로 쓰고 새로 고르지 않는다(후보 1e-4·5e-5·2e-5, seed 0, val Macro F1).
#   sup·pl·plc·ceil : 1e-4 — plots/lr_selected.json(2026-09-24), EfficientAT·PANNs·CED 셋 다 1e-4
#   표준 반지도     : 방법마다 — plots/lr_selected_<방법>.json(2026-10-08, EfficientAT 만), FreeMatch 만 5e-5
ADAM_LR = 1e-4
SSL_ADAM_LRS = {"pseudolabel": 1e-4, "meanteacher": 1e-4, "fixmatch": 1e-4, "freematch": 5e-5}


def adam_lr(cond):
    """[신설 2026-10-25 / A안] 그 조건의 학습률(위 선정 기록 그대로). train.py 가 fit 에 넘기고 train_meta 에 적으며, 실행기가 재개 판정 때 대조한다."""
    return SSL_ADAM_LRS[cond] if cond in SSL_METHODS else ADAM_LR


def run_tag(model_tag, cond, lr, seed, smoke=False, labeled_n=None, unlabeled_n=None):
    # [2026-10-25 / A안] lr=None 또는 CUR 이면 지금 절차(Adam·선정값 고정) 회차(_adam). 숫자는 2026-10-21까지의 Adam 시절 산출물 이름을 다시 만들 때만 쓴다.
    tag = f"{model_tag}_{cond}_adam" if (lr is None or lr == CUR) else f"{model_tag}_{cond}_lr{float(lr):g}"
    if labeled_n is not None:
        tag += f"_l{int(labeled_n)}"
    if unlabeled_n is not None:
        tag += f"_u{int(unlabeled_n)}"
    return tag + f"_s{int(seed)}" + ("_smoke" if smoke else "")


# [2026-10-22 / 검증 결합기] plc = labeled + Mark5.3 검증 결합기의 한 라벨 가짜 정답(common/mark53_labels.py). pl(옛 합의 라벨)과 태그를 나눠
#   옛 합의 라벨 결과가 새 결과로 잘못 읽히지 않게 한다.
_TAG_RE = re.compile(r"^(?P<model>.+?)_(?P<cond>sup|plc|pl|ceil|" + "|".join(SSL_METHODS) + r")_(?:lr(?P<lr>[0-9.e+-]+)|(?P<cur>adam))(?:_l(?P<l>\d+))?(?:_u(?P<u>\d+))?_s(?P<s>\d+)(?P<smoke>_smoke)?$")


def parse_tag(tag):
    """[신설 2026-09-17] run_tag -> dict(model, cond, lr, labeled(None=전량), unlabeled(None=전량), seed, smoke). 못 읽으면 None.
    [2026-10-25 / A안] _adam 태그는 lr=None(지금 절차, 실제 학습률은 adam_lr(cond)). _pd(Prodigy, 2026-10-22~10-25) 태그는 None."""
    m = _TAG_RE.match(tag)
    if not m:
        return None
    return {"model": m.group("model"), "cond": m.group("cond"), "lr": None if m.group("cur") else float(m.group("lr")),
            "labeled": int(m.group("l")) if m.group("l") else None,
            "unlabeled": int(m.group("u")) if m.group("u") else None,
            "seed": int(m.group("s")), "smoke": bool(m.group("smoke"))}


def artifact_paths(root, tag):
    plots = os.path.join(root, "plots")
    return {
        "ckpt": os.path.join(root, "model", f"{tag}.pth"),
        "loss": os.path.join(plots, f"loss_history_{tag}.csv"),      # 완주 표식
        "curve": os.path.join(plots, f"loss_curve_{tag}.png"),
        "meta": os.path.join(plots, f"train_meta_{tag}.json"),
        "val": os.path.join(plots, f"performance_summary_{tag}_val.csv"),
        "test": os.path.join(plots, f"performance_summary_{tag}.csv"),
        "rule": os.path.join(plots, f"others_threshold_{tag}.json"),      # [2026-09-17] val 에서 고른 others 규칙(test 가 읽음)
        "pl": os.path.join(plots, f"pl_quality_{tag}.csv"),                # [2026-10-07] 표준 반지도: 에폭별 가짜 정답 품질·양(표 3)
    }
