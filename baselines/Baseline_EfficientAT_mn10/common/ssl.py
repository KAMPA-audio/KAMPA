# -*- coding: utf-8 -*-
"""표준 반지도 기준선 4종 — Pseudo-Label · Mean Teacher · FixMatch · FreeMatch.

[신설 2026-10-07 / 논문 표 1] 같은 mn10 백본·같은 데이터·같은 학습 예산에서 Mark5.3 과 비교하기 위한 기준선이다.
반지도 방법끼리는 같은 백본·같은 코드에서 다시 구현해 비교하는 것이 표준이다(FixMatch NeurIPS 2020 4절,
FlexMatch/TorchSSL NeurIPS 2021, USB NeurIPS 2022, Cances 외 EURASIP 2022 — paper-wiki wiki/concepts/ssl-comparison-protocol.md).

손실 식과 하이퍼파라미터는 USB(microsoft/Semi-supervised-learning) 공식 구현을 옮겼다.
    semilearn/algorithms/{pseudolabel,meanteacher,fixmatch,freematch}/*.py
    semilearn/algorithms/freematch/utils.py (FreeMatchThresholingHook)
    semilearn/core/criterions/consistency.py (consistency_loss)
    config/usb_audio/<method>/<method>_esc50_250_0.yaml (오디오 기본값)
        pseudolabel : p_cutoff 0.95 · ulb_loss_ratio 1.0 · unsup_warm_up 0.4 · ema_m 0.0
        meanteacher : ema_m 0.999 · ulb_loss_ratio 50 · unsup_warm_up 0.4
        fixmatch    : p_cutoff 0.95 · hard_label · ulb_loss_ratio 1.0 · ema_m 0.0 (워밍업 없음)
        freematch   : ema_p 0.999 · hard_label · ent_loss_ratio 0.001(코드 기본은 0.01, 오디오 설정 파일 값 0.001) · ema_m 0.0

우리 프로토콜에 맞춰 바꾼 것 (수진님 결정 2026-10-07: 결정 1·2·3 + 기본값 모두 추천안)
    - 배치: Mark5.3 처럼 정답 클립과 라벨 없는 클립을 한 데이터셋으로 섞어 배치 16 (USB 는 정답·라벨없음 배치 1:1).
      한 배치에 정답이 없으면 지도 손실 0, 라벨 없는 클립이 없으면 비지도 손실 0.
    - 증강: 약한 증강 = 없음(3초 고정 클립이라 USB 의 길이 자르기 random_subsample 도 원본과 같다),
      강한 증강 = 기존 SpecAugment(common/augment.py, Mark5.3 FM 부품과 같은 규칙). 정답 클립 CE 는 sup 처럼 SpecAugment 입력.
      Pseudo-Label 은 USB 처럼 약한 입력 하나로 가짜 정답과 손실을 함께 만든다.
      Mean Teacher 는 USB 처럼 선생 = 약한 입력, 학생 = 강한 입력.
    - 비지도 손실 램프업: 네 방법 모두 처음 2에폭 동안 선형으로 0 → 1 (Mark5.3 FM 과 같다). USB 는 PL·MT 만 '전체 반복의 40%'
      램프업이고 FixMatch·FreeMatch 는 없다 — USB 반복 수(102,400)가 우리 80에폭 조기 종료 구조와 달라 하나로 맞췄다.
      FreeMatch 공정성 손실도 같은 램프를 탄다.
    - 평가 모델: USB 오디오 설정대로 Mean Teacher 만 EMA 선생, 나머지는 학습 모델 그대로(ema_m 0.0).
    - EMA 감쇠 [2026-10-08]: Mean Teacher 선생(ema_m)·FreeMatch 문턱(ema_p)을 USB 0.999 -> 0.99. USB 0.999 는 102,400 걸음
      기준(평균 폭 약 1,000 걸음 = 학습의 1%)이고, 우리는 에폭당 140~250 걸음 · 조기 종료까지 수천 걸음이라 0.999 면 선생·문턱이
      학습의 7~30% 를 평균해 크게 뒤처진다. Mark5.3 FM 부품이 같은 이유로 0.99 를 쓰므로(vild_config.py m53_fm_ema·m53_fm_momentum)
      기준선에도 같은 규칙을 적용했다. USB 원래 값은 hparams 의 usb_original 에 남긴다.
    - BN 통계 [2026-10-08 수정]: 학습 모델의 순전파는 네 방법 모두 섞은 배치 16개 전체를 한 번에 한다(BN 이 16개로 계산된다).
      USB 의 Pseudo-Label·Mean Teacher 는 정답 배치(8개)를 따로 순전파하고 라벨 없는 순전파에서 BN 을 얼리는데, 우리 섞은 배치에서
      그대로 하면 정답 라벨 200 일 때 정답이 배치당 약 1.5개(57% 배치가 0~1개)라 BN 이 1~2개 클립으로 계산된다(10-07 판의 결함:
      두 방법만 하한보다 낮았다). 정답 클립에만 SpecAugment 를 거는 Pseudo-Label 은 mel 을 만든 뒤 정답 클립에만 증강하고
      backbone 을 한 번 돈다(forward_mixed). Mean Teacher 선생(무증강, 라벨 없는 클립만)은 USB 처럼 BN 을 얼리고 순전파한다 —
      선생 버퍼는 매 걸음 학생 것으로 덮이므로 결과에 영향이 없다. FixMatch·FreeMatch 의 약한 입력 순전파(라벨 없는 클립만,
      no_grad)는 USB 처럼 얼리지 않는다.
    - 옵티마이저·조기 종료·학습률 후보는 비교모델 sup 과 같다(common/ssl_loop.py).
"""
import contextlib
import copy

import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from .augment import spec_augment
from .naming import SSL_METHODS, ssl_impl_version   # noqa: F401  (다른 모듈이 여기서 함께 가져간다)

USB_AUDIO_DEFAULTS = {
    "pseudolabel": {"p_cutoff": 0.95, "lambda_u": 1.0},
    "meanteacher": {"ema_m": 0.999, "lambda_u": 50.0},
    "fixmatch": {"p_cutoff": 0.95, "lambda_u": 1.0},
    "freematch": {"ema_p": 0.999, "lambda_u": 1.0, "lambda_e": 0.001},
}
# [2026-10-08] 우리 학습 길이에 맞춘 값(위 docstring 'EMA 감쇠'). USB 원래 값은 USB_AUDIO_DEFAULTS 그대로 두고 hparams 에 함께 적는다.
STEP_ADJUSTED = {"meanteacher": {"ema_m": 0.99}, "freematch": {"ema_p": 0.99}}
RAMP_EPOCHS = 2.0
EVAL_MODEL = {"pseudolabel": "student", "meanteacher": "ema_teacher", "fixmatch": "student", "freematch": "student"}


def hparams(method):
    """그 방법의 하이퍼파라미터 한 벌(로그·train_meta 에 그대로 적는다)."""
    if method not in SSL_METHODS:
        raise ValueError(f"[ERROR] 모르는 반지도 방법 {method!r}. 허용: {SSL_METHODS}")
    h = dict(USB_AUDIO_DEFAULTS[method])
    if method in STEP_ADJUSTED:
        h["usb_original"] = {k: USB_AUDIO_DEFAULTS[method][k] for k in STEP_ADJUSTED[method]}
        h.update(STEP_ADJUSTED[method])
    h["ramp_epochs"] = RAMP_EPOCHS
    h["eval_model"] = EVAL_MODEL[method]
    h["weak_aug"] = "none"
    h["strong_aug"] = "SpecAugment(common/augment.py)"
    return h


def forward_mixed(model, waves, aug_mask):
    """섞은 배치를 한 번에 순전파하되 aug_mask 가 True 인 클립에만 SpecAugment 를 건다(학습 모드일 때만 — model.forward 와 같은 규칙).
    BN 은 배치 전체로 계산된다. frontend(mel)는 클립마다 따로 계산되므로(배치에 의존하지 않음) 증강 위치만 model.forward 와 같다.
    [신설 2026-10-08] Pseudo-Label: 정답 = SpecAugment 입력, 라벨 없음 = 약한 입력(증강 없음)."""
    mel = model.frontend(waves)
    if model.training and bool(aug_mask.any()):
        mel = mel.clone()
        mel[aug_mask] = spec_augment(mel[aug_mask])
    return model.backbone(mel)


def ramp_weight(progress_epochs, ramp_epochs=RAMP_EPOCHS):
    """진행 에폭(소수 포함) -> 0~1. ramp_epochs <= 0 이면 처음부터 1."""
    if ramp_epochs <= 0:
        return 1.0
    return float(min(1.0, max(0.0, progress_epochs / ramp_epochs)))


@contextlib.contextmanager
def frozen_bn_stats(model):
    """USB core/utils/misc.py Bn_Controller(freeze_bn·unfreeze_bn)와 같다: 순전파 동안 바뀐 BN running 통계를 되돌린다
    (그 순전파 자체는 train 모드 배치 통계로 정규화된다). USB 의 Pseudo-Label·Mean Teacher 가 라벨 없는 클립을 순전파할 때 쓴다.
    [2026-10-08] 우리 구현에서는 Mean Teacher 선생의 순전파에만 쓴다(학생은 섞은 배치 전체를 한 번에 순전파).
    [2026-10-07 검수 반영 — code-reviewer low 1]"""
    backup = {}
    for name, m in model.named_modules():
        if isinstance(m, _BatchNorm) and m.track_running_stats and m.running_mean is not None:
            backup[name] = (m.running_mean.detach().clone(), m.running_var.detach().clone(), m.num_batches_tracked.detach().clone())
    try:
        yield
    finally:
        for name, m in model.named_modules():
            if name in backup:
                rm, rv, nb = backup[name]
                m.running_mean.data.copy_(rm)
                m.running_var.data.copy_(rv)
                m.num_batches_tracked.data.copy_(nb)


def ce_masked_mean(logits, targets, mask):
    """USB consistency_loss(logits, targets, 'ce', mask): 클립별 CE × mask 의 평균(분모 = 그 배치의 라벨 없는 클립 수)."""
    return (F.cross_entropy(logits, targets, reduction="none") * mask).mean()


def mse_consistency(logits_s, probs_t):
    """USB consistency_loss(logits, targets, 'mse'): softmax(학생) 과 선생 확률의 MSE 를 클래스 평균 → 클립 평균."""
    return F.mse_loss(torch.softmax(logits_s, dim=-1), probs_t, reduction="none").mean(dim=1).mean()


def _replace_inf_to_zero(val):
    val[val == float("inf")] = 0.0
    return val


def freematch_entropy_loss(mask, logits_s, prob_model, label_hist):
    """USB freematch.py entropy_loss 를 그대로 옮김(자기 적응 공정성 SAF 손실). 반환: (손실, hist_s 평균)."""
    mask = mask.bool()
    logits_s = logits_s[mask]
    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)
    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum()
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    prob_model_scaler = _replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)
    mean_prob_scaler_s = _replace_inf_to_zero(1 / hist_s).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)
    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()


class FreeMatchState:
    """USB freematch/utils.py FreeMatchThresholingHook 을 옮김(분산 학습 부분만 뺐다).
    전체 문턱 time_p(최대 확률 평균의 EMA) × 클래스별 비율 p_model/max(p_model) 로 클래스마다 문턱을 정한다."""

    def __init__(self, num_classes, momentum=0.999, device="cpu"):
        self.m = float(momentum)
        self.p_model = torch.ones(num_classes, device=device) / num_classes
        self.label_hist = torch.ones(num_classes, device=device) / num_classes
        self.time_p = self.p_model.mean()

    @torch.no_grad()
    def update(self, probs):
        max_probs, max_idx = torch.max(probs, dim=-1, keepdim=True)
        self.time_p = self.time_p * self.m + (1 - self.m) * max_probs.mean()
        self.p_model = self.p_model * self.m + (1 - self.m) * probs.mean(dim=0)
        hist = torch.bincount(max_idx.reshape(-1), minlength=self.p_model.shape[0]).to(self.p_model.dtype)
        self.label_hist = self.label_hist * self.m + (1 - self.m) * (hist / hist.sum())

    @torch.no_grad()
    def masking(self, probs, update=True):
        """USB 순서 그대로: 먼저 이번 배치 확률로 상태를 갱신하고, 갱신된 문턱으로 mask 를 만든다.
        update=False 는 학습이 끝난 뒤 저장된 상태로 가짜 정답 품질만 잴 때 쓴다."""
        if update:
            self.update(probs)
        max_probs, max_idx = probs.max(dim=-1)
        mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
        return max_probs.ge(self.time_p * mod[max_idx]).to(max_probs.dtype)

    def state_dict(self):
        return {"m": self.m, "p_model": self.p_model.detach().cpu().clone(),
                "label_hist": self.label_hist.detach().cpu().clone(), "time_p": float(self.time_p)}

    def load_state_dict(self, sd, device="cpu"):
        self.m = float(sd["m"])
        self.p_model = sd["p_model"].to(device)
        self.label_hist = sd["label_hist"].to(device)
        self.time_p = torch.as_tensor(float(sd["time_p"]), device=device)


class EMATeacher:
    """Mean Teacher 의 선생. 파라미터는 지수이동평균, 버퍼(BN 통계 등)는 학생 것을 그대로 가져온다 —
    USB 의 EMA(apply_shadow)가 파라미터만 바꿔 끼우고 BN 버퍼는 학생 것을 쓰는 것과 같다. 학습 중 선생 순전파는 USB 처럼
    train 모드(배치 통계)이고, 그때 바뀐 BN 통계는 다음 update 에서 학생 것으로 덮인다."""

    def __init__(self, student, decay):
        self.decay = float(decay)
        self.model = copy.deepcopy(student)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student):
        for pt, ps in zip(self.model.parameters(), student.parameters()):
            pt.mul_(self.decay).add_(ps.detach(), alpha=1.0 - self.decay)
        for bt, bs in zip(self.model.buffers(), student.buffers()):
            bt.copy_(bs)


class SSLObjective:
    """한 배치(정답·라벨 없음이 섞인 배치, 라벨 없음 = y < 0)의 손실을 방법별로 만든다.
    loss() -> (전체 손실, 기록용 값). 기록용 값의 pseudo·mask·unl_pos 는 가짜 정답 품질(표 3)을 재는 데만 쓰고 손실에는 쓰지 않는다."""

    def __init__(self, method, model, num_classes, device):
        self.method = method
        self.h = hparams(method)
        self.num_classes = num_classes
        self.teacher = EMATeacher(model, self.h["ema_m"]) if method == "meanteacher" else None
        self.fm = FreeMatchState(num_classes, self.h["ema_p"], device) if method == "freematch" else None

    def eval_model(self, model):
        """val·체크포인트·평가에 쓰는 모델(MT = EMA 선생, 나머지 = 학습 모델)."""
        return self.teacher.model if self.teacher is not None else model

    def after_step(self, model):
        if self.teacher is not None:
            self.teacher.update(model)

    def loss(self, model, waves, y, progress_epochs):
        lab = y >= 0
        unl = ~lab
        n_l, n_u = int(lab.sum().item()), int(unl.sum().item())
        zero = waves.new_zeros(())
        sup, unsup, ent = zero, zero, zero
        pseudo, mask = None, None
        if self.method == "pseudolabel":
            # [2026-10-08] 배치 전체 한 번 순전파(BN 16개). 정답 = SpecAugment 입력 CE, 라벨 없음 = 약한 입력(증강 없음) 하나로
            # 가짜 정답과 손실을 함께 만든다(USB: x_ulb_w). 옛 판은 정답만 따로 순전파해 BN 이 정답 1~2개로 계산됐다.
            logits = forward_mixed(model, waves, lab)
            if n_l:
                sup = F.cross_entropy(logits[lab], y[lab])
            if n_u:
                logits_u = logits[unl]
                probs = torch.softmax(logits_u.detach(), dim=-1)
                max_p, pseudo = probs.max(dim=-1)
                mask = max_p.ge(self.h["p_cutoff"]).to(max_p.dtype)
                unsup = ce_masked_mean(logits_u, pseudo, mask)
        elif self.method == "meanteacher":
            # [2026-10-08] 학생은 배치 전체 한 번 순전파(BN 16개): 정답 = SpecAugment 입력 CE, 라벨 없음 = 강한 입력(SpecAugment).
            # 옛 판은 USB 처럼 정답만 따로 순전파하고 라벨 없는 학생 순전파에서 BN 을 얼려, BN 이 정답 1~2개로 계산됐다.
            # 선생은 USB 처럼 train 모드·BN 얼림으로 라벨 없는 클립의 약한 입력만 본다(선생 버퍼는 update 때 학생 것으로 덮인다).
            # (USB 코드는 학생 강한 입력의 출력 자리에 outs_x_ulb_w 를 넣고 있다 — 오기로 보고, Mean Teacher 정의대로 학생의 강한 입력 출력을 쓴다.)
            logits = model(waves, augment=True)
            if n_l:
                sup = F.cross_entropy(logits[lab], y[lab])
            if n_u:
                with torch.no_grad(), frozen_bn_stats(self.teacher.model):
                    self.teacher.model.train()
                    logits_w = self.teacher.model(waves[unl], augment=False)
                logits_s = logits[unl]
                probs_w = torch.softmax(logits_w.detach(), dim=-1)
                unsup = mse_consistency(logits_s, probs_w)
                pseudo = probs_w.argmax(dim=-1)
                mask = torch.ones_like(probs_w[:, 0])                      # 문턱이 없다 — 양 100%
        else:
            logits = model(waves, augment=True)                            # 정답 = SpecAugment 입력 CE · 라벨 없음 = 강한 입력
            if n_l:
                sup = F.cross_entropy(logits[lab], y[lab])
            if n_u:
                logits_s = logits[unl]
                with torch.no_grad():                                      # FixMatch·FreeMatch: USB 처럼 BN 을 얼리지 않는다
                    logits_w = model(waves[unl], augment=False)
                probs_w = torch.softmax(logits_w.detach(), dim=-1)
                if self.method == "fixmatch":
                    max_p, pseudo = probs_w.max(dim=-1)
                    mask = max_p.ge(self.h["p_cutoff"]).to(max_p.dtype)
                    unsup = ce_masked_mean(logits_s, pseudo, mask)
                elif self.method == "freematch":
                    mask = self.fm.masking(probs_w, update=True)
                    pseudo = probs_w.argmax(dim=-1)
                    unsup = ce_masked_mean(logits_s, pseudo, mask)
                    if mask.sum() > 0:
                        ent, _ = freematch_entropy_loss(mask, logits_s, self.fm.p_model, self.fm.label_hist)
                else:
                    raise ValueError(self.method)
        w = ramp_weight(progress_epochs, self.h["ramp_epochs"])
        total = sup + w * (self.h["lambda_u"] * unsup + self.h.get("lambda_e", 0.0) * ent)
        stats = {"sup": float(sup.detach()), "unsup": float(unsup.detach()), "ent": float(torch.as_tensor(ent).detach()),
                 "ramp": w, "n_lab": n_l, "n_unl": n_u,
                 "pseudo": (pseudo.detach().cpu() if pseudo is not None else None),
                 "mask": (mask.detach().cpu() if mask is not None else None),
                 "unl_pos": torch.nonzero(unl, as_tuple=False).reshape(-1).cpu()}
        return total, stats

    @torch.no_grad()
    def pseudo_and_mask(self, eval_model, waves):
        """학습이 끝난 뒤 가짜 정답 품질(표 3)을 잴 때: 평가 모델(eval 모드)·증강 없음 입력으로 가짜 정답과 mask 를 만든다.
        FreeMatch 는 저장된 문턱 상태를 갱신하지 않고 쓴다. MT 는 문턱이 없어 mask = 1."""
        probs = torch.softmax(eval_model(waves, augment=False), dim=-1)
        max_p, pseudo = probs.max(dim=-1)
        if self.method in ("pseudolabel", "fixmatch"):
            mask = max_p.ge(self.h["p_cutoff"]).to(max_p.dtype)
        elif self.method == "freematch":
            mask = self.fm.masking(probs, update=False)
        else:
            mask = torch.ones_like(max_p)
        return pseudo, mask
