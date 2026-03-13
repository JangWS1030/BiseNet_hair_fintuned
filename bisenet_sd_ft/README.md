# BiSeNet SD Hair Fine-Tuning Workspace

Stable Diffusion 1.5 인페인팅 파이프라인용 헤어 세그멘테이션 모델을 별도 워크스페이스에서 파인튜닝하기 위한 독립 프로젝트입니다.

현재 서비스 컨텍스트:

- 얼굴 검출: MediaPipe FaceDetection
- 헤어 마스크 초안: 기존 BiSeNet `seg.pth`
- 마스크 정밀화: SAM2 `sam2.pt`
- 생성: `runwayml/stable-diffusion-inpainting`
- 구조 보존: `lllyasviel/control_v11p_sd15_canny`
- 얼굴 동일성 유지: `ip-adapter-plus-face_sd15.bin`

이번 프로젝트의 목표:

- 기존 BiSeNet 구조를 유지한 16-class 출력 모델로 파인튜닝
- coarse parsing 라벨 설계 유지
  - background `0`
  - face `1`
  - hair `10`
  - ignore `255`
- 기존 `seg.pth`를 초기 가중치로 로드 가능
- 최종 산출물 `seg_sd_ft.pth` 생성

## 폴더 구조

```text
bisenet_sd_ft/
├─ .env.example
├─ configs/
│  └─ bisenet_ft.yaml
├─ scripts/
│  ├─ download_aihub85.py
│  ├─ runpod_bootstrap.sh
│  ├─ prepare_aihub85.py
│  ├─ make_splits.py
│  └─ eval_external.py
├─ tools/
│  └─ aihubshell
├─ src/
│  ├─ models/
│  ├─ datasets/
│  ├─ losses/
│  ├─ train.py
│  ├─ eval.py
│  └─ infer.py
├─ outputs/
├─ requirements.txt
└─ README.md
```

## 현재 가정 사항

- AIHub 85 공개 페이지 설명 기준으로 `polygon1`은 헤어, `polygon2`는 얼굴로 간주합니다.
- `annotation.csv`, `image.csv`, `attribute.csv`, `meta-annotation.csv`가 라벨 패키지 내부에 존재한다고 가정합니다.
- 비주얼 서비스 파이프라인과 맞추기 위해 coarse parsing은 hair=`10`을 그대로 유지합니다.
- 미표기 영역은 현재 `background=0`으로 둡니다.
  `ignore=255`는 런타임 패딩/무효 영역용으로 예약합니다.
- 외부 평가용 AIHub 83은 데이터셋 key/구조가 현재 워크스페이스에서 확정되지 않았습니다.
  그래서 `scripts/eval_external.py`는 “AIHub 83도 같은 prepared manifest 포맷으로 변환됐다”는 전제로 훅만 우선 제공합니다.

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`.env`는 커밋하지 않고, `.env.example`을 복사해서 씁니다.

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Runpod에서 큰 데이터셋 다루는 권장 방식

대용량 데이터는 로컬에서 업로드하지 말고 Runpod 인스턴스 안에서 직접 내려받는 쪽이 안정적입니다.

권장 흐름:

1. Runpod `RTX 4090` Pod 생성
2. `Network Volume`을 충분히 크게 붙이기
3. 이 워크스페이스를 Pod 또는 Volume에 clone/copy
4. `.env` 또는 환경변수에 `AIHUB_APIKEY` 설정
5. Pod 내부에서 AIHub 85를 직접 다운로드
6. Pod 내부에서 전처리와 split 생성
7. 같은 Volume에서 학습 실행

Runpod 스토리지 선택:

- `Container Disk`: `20GB`
- `Volume Disk`: `0GB` 또는 최소치
- `Network Volume`: `150GB` 권장, 여유 있으면 `200GB`

어느 걸 고르냐면:

- 왼쪽 `Volume Disk`가 아니라 오른쪽 `Network Volume`을 사용하세요.
- `Volume Disk`는 Pod terminate 시 사실상 같이 사라지는 쪽이라 이번 용도에 맞지 않습니다.
- `Network Volume`은 `/workspace`에 붙고, Pod를 terminate해도 남아서 resume와 비용 절약에 유리합니다.

추천 저장공간:

- `hq` 프로필만 쓸 경우: 최소 150GB, 안전하게는 200GB
- `hq + mq`: 300GB 이상 권장
- `full`: 600GB 이상 권장

이유:

- AIHub tar 다운로드
- tar 해제
- 내부 zip 유지/해제
- prepared 이미지/마스크 생성
- 체크포인트/로그 저장

## 데이터 다운로드

`.env` 또는 쉘 환경변수에서 `AIHUB_APIKEY`를 읽습니다.

기본 권장: 먼저 `hq`만 내려받아 빠르게 검증한 뒤, 필요 시 `hq_mq`로 확장합니다.

```bash
python scripts/download_aihub85.py --out-dir data/aihub85_raw --profile hq --stage all
```

이 프로젝트는 루트에 있던 `aihubshell` 없이도 동작하지만, 필요하면 `tools/aihubshell`로 같이 넣어두었습니다.

드라이런:

```bash
python scripts/download_aihub85.py --out-dir data/aihub85_raw --profile hq --stage all --dry-run
```

## 데이터 준비

```bash
python scripts/prepare_aihub85.py --raw-dir data/aihub85_raw --out-dir data/aihub85_prepared
```

품질 선택:

```bash
python scripts/prepare_aihub85.py --raw-dir data/aihub85_raw --out-dir data/aihub85_prepared --qualities hq mq
```

샘플 검증용 제한:

```bash
python scripts/prepare_aihub85.py --raw-dir data/aihub85_raw --out-dir data/aihub85_prepared --qualities hq --max-samples 500
```

출력 결과:

- `data/aihub85_prepared/images/*`
- `data/aihub85_prepared/masks/*`
- `data/aihub85_prepared/manifest.csv`
- `data/aihub85_prepared/prepare_summary.json`

## Split 생성

```bash
python scripts/make_splits.py --data-dir data/aihub85_prepared --seed 42
```

현재 split 로직:

- `80/10/10`
- `group_id -> subject_id -> sequence_id -> sample_id` 순으로 그룹 키 선택
- 같은 그룹은 같은 split에 유지
- `short_hair`, `bangs`, `sideburn`, `dark_hair`가 있으면 분포 균형을 맞추도록 greedy 배치

## 초기 가중치 설정

`configs/bisenet_ft.yaml`의 `model.init_checkpoint`에 기존 `seg.pth` 경로를 넣으면 됩니다.

예시:

```yaml
model:
  init_checkpoint: /workspace/checkpoints/seg.pth
  strict_load: true
  allow_partial_load: true
```

체크포인트를 넣지 않으면 랜덤 초기화 + ResNet18 backbone pretrained로 시작합니다.

## 학습

```bash
python src/train.py --config configs/bisenet_ft.yaml
```

학습 최적화 포인트:

- AMP `bf16`
- gradient accumulation
- `channels_last`
- cosine LR + warmup
- boundary-aware loss 포함
- 16-class 출력 유지
- hair=`10` 유지

기본 loss 조합:

```text
total = CE
      + 0.4 * aux_CE_16
      + 0.4 * aux_CE_32
      + 0.5 * Dice(hair)
      + 0.2 * Dice(face)
      + 0.2 * Boundary(hair)
```

출력:

- `outputs/<experiment>/best.pth`
- `outputs/<experiment>/latest.pth`
- `outputs/<experiment>/seg_sd_ft.pth`
- `outputs/<experiment>/history.csv`
- `outputs/<experiment>/metrics_epoch_*.json`

재시작:

`latest.pth` 또는 `best.pth`에서 이어서 학습하려면 config에 아래를 넣습니다.

```yaml
model:
  resume_checkpoint: /workspace/bisenet_sd_ft/outputs/aihub85_hair_sd_ft/latest.pth
```

## 평가

```bash
python src/eval.py --config configs/bisenet_ft.yaml --checkpoint outputs/aihub85_hair_sd_ft/best.pth
```

테스트 split 평가:

```bash
python src/eval.py --config configs/bisenet_ft.yaml --checkpoint outputs/aihub85_hair_sd_ft/best.pth --split test
```

기본 지표:

- Hair IoU
- Hair Dice
- Boundary F1
- Face spill rate
- Face IoU

subset metric:

- `short_hair`
- `bangs`
- `sideburn`
- `dark_hair`

## 외부 평가 훅

외부 데이터셋을 같은 prepared manifest 형식으로 만든 뒤 평가할 수 있습니다.

```bash
python scripts/eval_external.py \
  --config configs/bisenet_ft.yaml \
  --checkpoint outputs/aihub85_hair_sd_ft/best.pth \
  --manifest data/aihub83_prepared/manifest.csv \
  --split-file data/aihub83_prepared/splits/external.txt \
  --save-dir outputs/aihub83_eval
```

## 단일 이미지 추론

```bash
python src/infer.py --checkpoint outputs/aihub85_hair_sd_ft/seg_sd_ft.pth --image sample.jpg --out outputs/infer/sample_mask.png
```

저장 파일:

- `sample_mask.png`: label map
- `sample_mask_hair.png`: hair binary mask
- `sample_mask_overlay.png`: overlay preview

## 필수 CLI 예시

데이터 준비:

```bash
python scripts/prepare_aihub85.py --raw-dir ... --out-dir ...
```

split 생성:

```bash
python scripts/make_splits.py --data-dir ... --seed 42
```

학습:

```bash
python src/train.py --config configs/bisenet_ft.yaml
```

평가:

```bash
python src/eval.py --config configs/bisenet_ft.yaml --checkpoint outputs/.../best.pth
```

추론:

```bash
python src/infer.py --checkpoint ... --image ... --out ...
```

## Runpod 운영 팁

- 처음에는 `hq` 프로필로 1회 학습해 loss/metric과 마스크 품질을 확인합니다.
- 그 다음 `hq_mq`로 넓혀서 일반화 성능을 올립니다.
- 체크포인트와 prepared data는 항상 `Network Volume`에 두고, Pod는 교체 가능한 계산 노드로 취급하는 편이 안전합니다.
- 긴 학습은 `tmux`, `screen`, 또는 `nohup`으로 분리 실행하세요.
- 최종 배포용 파일은 `outputs/<run>/seg_sd_ft.pth`만 서비스로 넘기면 됩니다.

퇴근 전 추천 방식:

```bash
bash scripts/runpod_train.sh configs/bisenet_ft.yaml 3 terminate_on_success
```

이 스크립트는:

- `nohup`으로 SSH 종료와 무관하게 학습 지속
- 로그를 `outputs/launcher_logs/*.log`에 저장
- 학습 성공 시 자동 terminate 가능
- 선택적으로 일정 시간 뒤 Pod terminate 예약

주의:

- `Network Volume`을 붙인 Pod라면 데이터는 `/workspace`에 남습니다.
- auto terminate는 Pod를 지우는 방식이라 container disk 내용은 사라집니다.
- 따라서 체크포인트와 데이터는 반드시 `/workspace` 아래 Volume에 있어야 합니다.

자동 종료 모드:

- `none`: 학습 완료 후 Pod 유지
- `terminate_on_success`: 학습이 정상 종료되면 terminate
- `terminate_on_exit`: 에러 여부와 무관하게 terminate
- `stop_on_success`: 학습이 정상 종료되면 stop
- `stop_on_exit`: 에러 여부와 무관하게 stop

예시:

```bash
bash scripts/runpod_train.sh configs/bisenet_ft.yaml "" terminate_on_success
```

시간 기반 종료만 걸기:

```bash
bash scripts/runpod_train.sh configs/bisenet_ft.yaml 3 none
```

둘 다 같이 쓰기:

```bash
bash scripts/runpod_train.sh configs/bisenet_ft.yaml 3 terminate_on_success
```

Volume Disk만 쓸 때 권장:

```bash
bash scripts/runpod_train.sh configs/bisenet_ft.yaml 8 stop_on_success
```

## GitHub + Runpod 빠른 시작

이 프로젝트는 `bisenet_sd_ft` 폴더만 별도 저장소로 올려서 쓰는 걸 권장합니다.

Runpod에서의 순서:

```bash
git clone <your-repo-url>
cd bisenet_sd_ft
bash scripts/runpod_bootstrap.sh
cp .env.example .env
# .env에 AIHUB_APIKEY 입력
python scripts/download_aihub85.py --out-dir data/aihub85_raw --profile hq --stage all
python scripts/prepare_aihub85.py --raw-dir data/aihub85_raw --out-dir data/aihub85_prepared
python scripts/make_splits.py --data-dir data/aihub85_prepared --seed 42
# configs/bisenet_ft.yaml 에 init_checkpoint 경로 설정
bash scripts/runpod_train.sh configs/bisenet_ft.yaml 6 terminate_on_success
```

`seg.pth`를 같이 쓸 거면 아래 둘 중 하나로 가져가면 됩니다.

- 깃헙에 올리지 않고 Runpod에 직접 업로드
- Hugging Face private repo, Google Drive, S3 같은 외부 저장소에서 내려받기
