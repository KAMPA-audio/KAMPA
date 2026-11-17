# 출처
- 저장소: https://github.com/qiuqiangkong/audioset_tagging_cnn (MIT)
- 커밋: d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4  (2026-09-05 clone)
- 가져온 파일: pytorch/models.py, pytorch/pytorch_utils.py  — **수정 없음(원본 그대로)**
- 함께 필요한 패키지: torchlibrosa (pip, MIT) — models.py 가 STFT·log-mel 을 여기서 가져온다
- 가중치: weights/Cnn6_mAP=0.343.pth  <- https://zenodo.org/record/3987831/files/Cnn6_mAP%3D0.343.pth
- 우리가 쓰는 방식은 ../../model_def.py 참조 (Cnn6 부품을 같은 순서로 다시 잇고, 헤드만 9-class fc 로 교체)
