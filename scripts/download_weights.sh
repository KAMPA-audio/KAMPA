#!/usr/bin/env bash
# Download the released weights into the folders the code reads, then check their SHA-256.
#   model/student_model_*.pth               40 student checkpoints (8 settings x seeds 1-5) behind the four result tables
#   resources/mn10_teacher_mark4.{1..8}.pt  8 binary mn10 teachers (resources/teacher_manifest_mn10.json)
#   resources/best_teacher_{encoder,classifier}_mark4.{1..8}.pth  16 files of the segment teachers (B0), the reference in the
#                                           teacher gate (resources/teacher_manifest.json)
#   weights/mn10_as_mAP_471.pt              EfficientAT mn10 AudioSet weights (third party, EfficientAT release v0.0.1)
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${KAMPA_RELEASE:-v1.0.0}"
BASE="https://github.com/KAMPA-audio/KAMPA/releases/download/$TAG"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
for asset in kampa_student_checkpoints.zip kampa_mn10_teachers.zip kampa_segment_teachers.zip SHA256SUMS; do
  curl -fL --retry 3 -o "$tmp/$asset" "$BASE/$asset"
done
unzip -o -q "$tmp/kampa_student_checkpoints.zip" -d .
unzip -o -q "$tmp/kampa_mn10_teachers.zip" -d .
unzip -o -q "$tmp/kampa_segment_teachers.zip" -d .
mkdir -p weights
if [ ! -f weights/mn10_as_mAP_471.pt ]; then
  curl -fL --retry 3 -o weights/mn10_as_mAP_471.pt \
    https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt
fi
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum -c --quiet "$tmp/SHA256SUMS"
else
  shasum -a 256 -c --quiet "$tmp/SHA256SUMS"
fi
echo "[OK] 40 student checkpoints, 8 mn10 teachers, 16 segment-teacher files and the EfficientAT mn10 weights are in place."
