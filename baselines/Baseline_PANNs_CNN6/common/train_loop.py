# -*- coding: utf-8 -*-
"""파인튜닝 루프 — Mark5_EXP train_mark5.py 의 학습 골격(2026-08-17 판)을 그대로 따른다.

[신설 2026-09-05] 조건표 2절 '최적화'.

    Adam(lr, weight_decay=1e-4) · ReduceLROnPlateau(mode=min, factor 0.5, patience 3)
    [변경 2026-10-25 / A안] 2026-10-22~10-25 의 Prodigy(자동 학습률)를 걷어내고 Adam 으로 되돌렸다(보폭이 커져 사전학습 가중치가 지워졌다).
    lr 은 train.py 가 common/naming.adam_lr(cond)(Adam 시절 선정값 고정)로 넘긴다. 학습률이 줄어든 에폭의 [LR] 한 줄은 남긴다.
    · early stopping patience 10 · val 손실이 좋아질 때마다 체크포인트 저장(best-val)
    · 최대 80 epoch · 손실 = cross-entropy(hard label)

학습률만 모델마다 val 로 고른다(조건표 5절). SpecAugment 는 학습 배치에만 건다.

산출물 두 가지의 역할이 다르다(Mark5_EXP run_experiments.py 의 재개 규칙과 같다):
    ckpt_path      : val 이 좋아질 때마다 에폭 루프 **안에서** 덮어쓴다 -> 중도본일 수 있다.
    loss_csv_path  : 에폭 루프가 **끝난 뒤**에만 쓴다 -> 이 파일이 '완주 표식'이다.
둘 다 있어야 학습이 끝난 것으로 본다.
"""
import csv
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _run_epoch_eval(model, loader, device):
    """val 손실은 Mark5(train_mark5.py:835 `val_total / len(val_loader)`)와 같은 **배치 평균의 평균**으로 둔다.
    표본 평균과 마지막 배치(430 % 16 = 6개)에서만 미세하게 다르지만, early stopping·스케줄러 판정이 경계에서
    갈리지 않게 골격을 그대로 맞춘다."""
    model.eval()
    n_batches, correct, total, loss_sum = 0, 0, 0, 0.0
    with torch.no_grad():
        for waves, y, _ in loader:
            waves, y = waves.to(device), y.to(device)
            logits = model(waves, augment=False)
            loss_sum += F.cross_entropy(logits, y).item()
            n_batches += 1
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
    return (loss_sum / max(n_batches, 1)), (correct / max(total, 1))


def fit(model, train_loader, val_loader, device, *, lr, weight_decay=1e-4, epochs=80, patience=10,
        sched_factor=0.5, sched_patience=3, use_spec_augment=True,
        ckpt_path, loss_csv_path, curve_png_path=None, tag="", extra_ckpt_meta=None):
    # [2026-09-05 저녁, code-reviewer 지적] 옛 산출물을 먼저 지운다. 지우지 않으면 --force 재실행이 도중에 죽었을 때
    # 새 실행의 중도 체크포인트 + 옛 완주 표식 조합이 남아, 다음 재개에서 덜 배운 모델을 완료로 착각한다.
    for stale in (ckpt_path, loss_csv_path):
        if stale and os.path.exists(stale):
            os.remove(stale)
            print(f"[INFO] 옛 산출물 삭제: {stale}")
    if lr is None or float(lr) <= 0:   # [2026-10-25 / A안] 학습률은 train.py 가 adam_lr(cond) 로 넘긴다
        raise ValueError(f"[ERROR] 학습률이 없거나 양수가 아닙니다(받은 값 {lr}).")
    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=float(lr), weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=sched_factor, patience=sched_patience)
    print(f"[INFO] 학습 절차: Adam lr={float(lr):g}, weight_decay={weight_decay}, "
          f"ReduceLROnPlateau(factor={sched_factor}, patience={sched_patience}), "
          f"early stopping patience={patience}, best val(loss) 체크포인트 저장, "
          f"SpecAugment={'on' if use_spec_augment else 'off'}, 최대 {epochs} epoch")

    best_val, best_epoch, no_improve, stopped_early = math.inf, -1, 0, False
    history = []
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        loss_sum, n_batches, correct, total = 0.0, 0, 0, 0
        for waves, y, _ in train_loader:
            waves, y = waves.to(device), y.to(device)
            logits = model(waves, augment=use_spec_augment)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
        train_loss, train_acc = loss_sum / max(n_batches, 1), correct / max(total, 1)   # Mark5 와 같은 배치 평균
        val_loss, val_acc = _run_epoch_eval(model, val_loader, device)
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "lr": cur_lr, "seconds": round(elapsed, 1)})
        print(f"[Epoch {epoch + 1}] Train {train_loss:.4f} (acc {train_acc:.4f}) | "
              f"Val {val_loss:.4f} (acc {val_acc:.4f}) | lr {cur_lr:g} | {elapsed:.1f}s", flush=True)

        scheduler.step(val_loss)
        if optimizer.param_groups[0]["lr"] < cur_lr:   # [2026-10-22] Mark5.3 과 같은 [LR] 한 줄
            print(f"[LR] {cur_lr:g} -> {optimizer.param_groups[0]['lr']:g} (val={val_loss:.6f})")
        if val_loss < best_val:
            print(f"[EarlyStopping] Val loss {best_val:.6f} -> {val_loss:.6f}. Saving...")
            best_val, best_epoch, no_improve = val_loss, epoch + 1, 0
            os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
            payload = {"model_state": model.state_dict(), "epoch": epoch + 1, "val_loss": val_loss,
                       "val_acc": val_acc, "lr": lr, "optimizer": "adam", "tag": tag}
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
        print("[ERROR] val 이 한 번도 좋아지지 않아 체크포인트가 없습니다(val loss 가 NaN 인지 확인). "
              "손실 곡선만 저장합니다.")

    # 완주 표식. 루프가 끝난 뒤에만 쓴다.
    os.makedirs(os.path.dirname(os.path.abspath(loss_csv_path)), exist_ok=True)
    with open(loss_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"])
        w.writeheader()
        w.writerows(history)
    print(f"[INFO] 손실 곡선 CSV 저장(완주 표식): {loss_csv_path}")

    if curve_png_path and history:
        plt.figure(figsize=(8, 5))
        plt.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="Train CE", linewidth=2)
        plt.plot([h["epoch"] for h in history], [h["val_loss"] for h in history], label="Val CE", linewidth=2)
        if best_epoch > 0:
            plt.axvline(best_epoch, color="gray", linestyle=":", label=f"best epoch {best_epoch}")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"Loss ({tag})")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(curve_png_path); plt.close()

    return {"best_epoch": best_epoch, "best_val_loss": best_val, "epochs_run": len(history),
            "stopped_early": stopped_early, "history": history}
