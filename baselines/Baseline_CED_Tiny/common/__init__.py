# common — 비교모델 세 폴더(Baseline_PANNs_CNN6 · Baseline_EfficientAT_mn10 · Baseline_CED_Tiny)가
# 똑같이 쓰는 공통 코드. 세 폴더에 같은 사본이 들어 있다(폴더마다 독립 저장소라서).
# 한 곳을 고치면 나머지 두 곳도 같이 고친다. 조건의 근거는
# Downloads/Mark5_비교실험_조건표_20260905.pdf (2026-09-05).
# [2026-10-07] 예외: ssl.py · ssl_loop.py(표준 반지도 기준선, 논문 표 1)는 Baseline_EfficientAT_mn10 에만 있다 — 표 1 은 같은 mn10
# 에서만 비교하기 때문이다(사본이 아니다). naming.py 의 SSL_METHODS 등은 네 폴더(CLAP 포함)에 같은 사본으로 둔다.
