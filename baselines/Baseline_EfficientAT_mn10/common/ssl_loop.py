# -*- coding: utf-8 -*-
"""표준 반지도 기준선 학습 루프 — common/train_loop.fit 과 같은 골격에 섞은 배치·방법별 손실만 더한다.

[신설 2026-10-07 / 논문 표 1] 기존 fit()(sup·pl·ceil)은 건드리지 않는다.

    같은 것: Adam(lr, weight_decay=1e-4) · ReduceLROnPlateau(min, 0.5, 3) · early stopping patience 10 · 최대 80 epoch
             [변경 2026-10-25 / A안] 2026-10-22~10-25 의 Prodigy 를 걷어내고 Adam 으로 되돌렸다 — train_loop.fit 과 같다.
             lr 은 train.py 가 common/naming.adam_lr(방법)(lr_selected_<방법>.json 선정값 고정)로 넘긴다.
             · val 손실(배치 평균의 평균)이 좋아질 때마다 체크포인트 저장 · 루프가 끝난 뒤에만 손실 CSV(완주 표식)
    다른 것: 배치 = 정답 클립 + 라벨 없는 클립을 섞은 배치(라벨 없음 = y < 0, Mark5.3 과 같은 구성)
             · 손실 = common/ssl.py SSLObjective · val·체크포인트는 평가 모델(MT = EMA 선생, 나머지 = 학습 모델)

체크포인트 "model_state" 는 평가 모델 가중치다 — eval.py 가 그대로 읽는다. MT 는 학습 모델도 "student_state" 로 함께 둔다.
FreeMatch 는 그때의 문턱 상태를 "ssl_state" 로 둔다(가짜 정답 품질을 잴 때 쓴다).

가짜 정답 품질·양(표 3, Grollmisch 외 EUSIPCO 2023 · FixMatch 부록 B.2 와 같은 두 값)
    품질 = mask 를 통과한 가짜 정답 중 숨긴 정답(ceiling_labels.csv)과 같은 비율, 양 = mask 를 통과한 비율.
    (1) 에폭마다: 그 에폭에 학습한 라벨 없는 클립 기준 -> plots/pl_quality_<tag>.csv
    (2) 학습이 끝난 뒤: 가장 좋은 체크포인트(평가 모델, eval 모드, 증강 없음)로 라벨 없는 클립 전부 -> train_meta 의 pl_quality_final
    숨긴 정답은 이 두 기록에만 쓰고 손실에는 절대 들어가지 않는다(손실은 y < 0 인 클립에 정답을 받지 않는다).
"""
import csv
import math
import os
import time

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import clip_key
from .ssl import SSLObjective, ssl_impl_version
from .train_loop import _run_epoch_eval

HIST_FIELDS = ["epoch", "train_loss", "sup_loss", "unsup_loss", "ent_loss", "ramp", "n_lab", "n_unl",
               "mask_rate", "pl_quality", "pl_n_masked", "val_loss", "val_acc", "lr", "seconds"]


def _pl_counts(stats, metas, gt_by_key):
    """한 배치의 가짜 정답 기록. 반환: (라벨 없는 클립 수, mask 통과 수, 그중 정답과 같은 수, 정답을 아는 통과 수)."""
    if stats["pseudo"] is None or stats["n_unl"] == 0:
        return 0, 0, 0, 0
    n_u, n_m, n_ok, n_known = stats["n_unl"], 0, 0, 0
    for j, pos in enumerate(stats["unl_pos"].tolist()):
        if float(stats["mask"][j]) <= 0:
            continue
        n_m += 1
        gt = gt_by_key.get(clip_key(metas[pos]["path"])) if gt_by_key else None
        if gt is None:
            continue
        n_known += 1
        n_ok += int(int(stats["pseudo"][j]) == int(gt))
    return n_u, n_m, n_ok, n_known


@torch.no_grad()
def pl_quality_pass(objective, eval_model, unl_loader, device, gt_by_key):
    """학습이 끝난 뒤 라벨 없는 클립 전부에 대해 가짜 정답 품질·양을 잰다(평가 모델 eval 모드, 증강 없음)."""
    eval_model.eval()
    n, n_m, n_ok, n_known, n_all_ok, n_all_known = 0, 0, 0, 0, 0, 0
    for waves, _y, metas in unl_loader:
        pseudo, mask = objective.pseudo_and_mask(eval_model, waves.to(device))
        for j, meta in enumerate(metas):
            n += 1
            gt = gt_by_key.get(clip_key(meta["path"])) if gt_by_key else None
            if gt is not None:
                n_all_known += 1
                n_all_ok += int(int(pseudo[j]) == int(gt))
            if float(mask[j]) > 0:
                n_m += 1
                if gt is not None:
                    n_known += 1
                    n_ok += int(int(pseudo[j]) == int(gt))
    # [2026-10-07 검수 반영 — code-reviewer low 2] 문턱을 넘은 클립이 0개면 품질은 정의되지 않는다 -> None(JSON null).
    # NaN 을 쓰면 train_meta 가 비표준 JSON 이 되고 표 3 평균이 조용히 nan 이 된다.
    return {"n_unlabeled": n, "n_masked": n_m, "quantity": (n_m / n) if n else None,
            "quality": (n_ok / n_known) if n_known else None, "n_masked_with_gt": n_known,
            "argmax_acc_all": (n_all_ok / n_all_known) if n_all_known else None, "n_with_gt": n_all_known}


def _fmt(v):
    return "없음" if v is None else f"{v:.4f}"


def fit_ssl(model, train_loader, val_loader, device, *, method, lr, weight_decay=1e-4, epochs=80, patience=10,
            sched_factor=0.5, sched_patience=3, num_classes=9,
            ckpt_path, loss_csv_path, curve_png_path=None, pl_csv_path=None, tag="", extra_ckpt_meta=None,
            gt_by_key=None, unl_eval_loader=None):
    for stale in (ckpt_path, loss_csv_path, pl_csv_path):
        if stale and os.path.exists(stale):
            os.remove(stale)
            print(f"[INFO] 옛 산출물 삭제: {stale}")
    objective = SSLObjective(method, model, num_classes, device)
    if lr is None or float(lr) <= 0:   # [2026-10-25 / A안] 학습률은 train.py 가 adam_lr(방법) 으로 넘긴다
        raise ValueError(f"[ERROR] 학습률이 없거나 양수가 아닙니다(받은 값 {lr}).")
    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=float(lr), weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=sched_factor, patience=sched_patience)
    print(f"[INFO] 학습 절차: 표준 반지도 {method} {objective.h} · Adam lr={float(lr):g}, weight_decay={weight_decay}, "
          f"ReduceLROnPlateau(factor={sched_factor}, patience={sched_patience}), early stopping patience={patience}, "
          f"best val(loss, 평가 모델={objective.h['eval_model']}) 체크포인트 저장, 최대 {epochs} epoch, "
          f"배치 = 정답 + 라벨 없는 클립 섞음({len(train_loader)} step/epoch)")
    if not gt_by_key:
        print("[WARN] 숨긴 정답(ceiling_labels.csv)이 없어 가짜 정답 품질은 기록하지 않습니다(양만 기록).")

    steps_per_epoch = max(len(train_loader), 1)
    best_val, best_epoch, no_improve, stopped_early = math.inf, -1, 0, False
    history = []
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        sums = {"loss": 0.0, "sup": 0.0, "unsup": 0.0, "ent": 0.0, "ramp": 0.0}
        n_batches, n_lab, n_unl, n_mask, n_ok, n_known = 0, 0, 0, 0, 0, 0
        for b_idx, (waves, y, metas) in enumerate(train_loader):
            waves, y = waves.to(device), y.to(device)
            loss, st = objective.loss(model, waves, y, epoch + b_idx / steps_per_epoch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            objective.after_step(model)
            sums["loss"] += float(loss.detach())
            for k in ("sup", "unsup", "ent", "ramp"):
                sums[k] += st[k]
            n_batches += 1
            n_lab += st["n_lab"]
            a, b, c, d = _pl_counts(st, metas, gt_by_key)
            n_unl += a
            n_mask += b
            n_ok += c
            n_known += d
        eval_m = objective.eval_model(model)
        val_loss, val_acc = _run_epoch_eval(eval_m, val_loader, device)
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        row = {"epoch": epoch + 1, "train_loss": sums["loss"] / max(n_batches, 1),
               "sup_loss": sums["sup"] / max(n_batches, 1), "unsup_loss": sums["unsup"] / max(n_batches, 1),
               "ent_loss": sums["ent"] / max(n_batches, 1), "ramp": sums["ramp"] / max(n_batches, 1),
               "n_lab": n_lab, "n_unl": n_unl, "mask_rate": (n_mask / n_unl) if n_unl else float("nan"),
               "pl_quality": (n_ok / n_known) if n_known else float("nan"), "pl_n_masked": n_mask,
               "val_loss": val_loss, "val_acc": val_acc, "lr": cur_lr, "seconds": round(elapsed, 1)}
        history.append(row)
        print(f"[Epoch {epoch + 1}] Train {row['train_loss']:.4f} (sup {row['sup_loss']:.4f} · unsup {row['unsup_loss']:.4f}"
              + (f" · ent {row['ent_loss']:.4f}" if method == "freematch" else "")
              + f" · ramp {row['ramp']:.2f}) | 가짜 정답 양 {row['mask_rate']:.3f} 품질 {row['pl_quality']:.3f} "
              f"({n_mask}/{n_unl}) | Val {val_loss:.4f} (acc {val_acc:.4f}) | lr {cur_lr:g} | {elapsed:.1f}s", flush=True)

        scheduler.step(val_loss)
        if optimizer.param_groups[0]["lr"] < cur_lr:   # [2026-10-22] Mark5.3 과 같은 [LR] 한 줄
            print(f"[LR] {cur_lr:g} -> {optimizer.param_groups[0]['lr']:g} (val={val_loss:.6f})")
        if val_loss < best_val:
            print(f"[EarlyStopping] Val loss {best_val:.6f} -> {val_loss:.6f}. Saving...")
            best_val, best_epoch, no_improve = val_loss, epoch + 1, 0
            os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
            payload = {"model_state": eval_m.state_dict(), "epoch": epoch + 1, "val_loss": val_loss,
                       "val_acc": val_acc, "lr": lr, "optimizer": "adam", "tag": tag, "ssl_method": method,
                       "eval_model": objective.h["eval_model"], "ssl_impl_version": ssl_impl_version(method)}
            if objective.teacher is not None:
                payload["student_state"] = model.state_dict()
            if objective.fm is not None:
                payload["ssl_state"] = objective.fm.state_dict()
            if extra_ckpt_meta:
                payload.update(extra_ckpt_meta)
            torch.save(payload, ckpt_path)
        else:
            no_improve += 1
            print(f"[EarlyStopping] counter: {no_improve}/{patience}")
            if no_improve >= patience:
                stopped_early = True
                print(f"[INFO] Early stopping. best epoch {best_epoch} (val {best_val:.6f})")
                break

    print("Training finished."
          + (f" (early stop, best epoch {best_epoch}/{epochs}, val {best_val:.6f})" if stopped_early
             else f" (best epoch {best_epoch}/{epochs}, val {best_val:.6f})"))
    if best_epoch < 0:
        print("[ERROR] val 이 한 번도 좋아지지 않아 체크포인트가 없습니다(val loss 가 NaN 인지 확인). 손실 곡선만 저장합니다.")

    # 가짜 정답 품질(학습 뒤, 가장 좋은 체크포인트) — 완주 표식보다 먼저 한다(여기서 멈추면 표식이 없어 다시 돈다).
    pl_final = None
    if unl_eval_loader is not None and best_epoch > 0:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        eval_m = objective.eval_model(model)
        eval_m.load_state_dict(ck["model_state"])
        if objective.fm is not None and "ssl_state" in ck:
            objective.fm.load_state_dict(ck["ssl_state"], device=device)
        pl_final = pl_quality_pass(objective, eval_m, unl_eval_loader, device, gt_by_key)
        pl_final["epoch"] = best_epoch
        print(f"[INFO] 가짜 정답(학습 뒤, best epoch {best_epoch}, 라벨 없는 클립 {pl_final['n_unlabeled']}개): "
              f"양 {_fmt(pl_final['quantity'])} · 품질 {_fmt(pl_final['quality'])} (문턱 통과 0개면 없음) · "
              f"argmax 전체 정확도 {_fmt(pl_final['argmax_acc_all'])}")

    if pl_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(pl_csv_path)), exist_ok=True)
        with open(pl_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "n_unl", "pl_n_masked", "mask_rate", "pl_quality", "ramp"])
            w.writeheader()
            for h in history:
                w.writerow({k: h[k] for k in ("epoch", "n_unl", "pl_n_masked", "mask_rate", "pl_quality", "ramp")})
        print(f"[INFO] 에폭별 가짜 정답 품질·양 CSV: {pl_csv_path}")

    # 완주 표식. 루프와 학습 뒤 기록이 모두 끝난 뒤에만 쓴다.
    os.makedirs(os.path.dirname(os.path.abspath(loss_csv_path)), exist_ok=True)
    with open(loss_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HIST_FIELDS)
        w.writeheader()
        w.writerows(history)
    print(f"[INFO] 손실 곡선 CSV 저장(완주 표식): {loss_csv_path}")

    if curve_png_path and history:
        plt.figure(figsize=(8, 5))
        ep = [h["epoch"] for h in history]
        plt.plot(ep, [h["train_loss"] for h in history], label="Train total", linewidth=2)
        plt.plot(ep, [h["sup_loss"] for h in history], label="Train sup CE", linewidth=1)
        plt.plot(ep, [h["val_loss"] for h in history], label="Val CE", linewidth=2)
        if best_epoch > 0:
            plt.axvline(best_epoch, color="gray", linestyle=":", label=f"best epoch {best_epoch}")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"Loss ({tag})")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(curve_png_path); plt.close()

    best_row = next((h for h in history if h["epoch"] == best_epoch), None)
    return {"best_epoch": best_epoch, "best_val_loss": best_val, "epochs_run": len(history),
            "stopped_early": stopped_early, "history": history, "pl_final": pl_final,
            "pl_best_epoch_train": ({k: (None if isinstance(best_row[k], float) and best_row[k] != best_row[k] else best_row[k])
                                     for k in ("mask_rate", "pl_quality", "pl_n_masked", "n_unl")}   # NaN -> None(JSON null)
                                    if best_row else None),
            "ssl_hparams": objective.h}
