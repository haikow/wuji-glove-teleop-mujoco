#!/usr/bin/env bash
# 数据飞轮跑一圈：质检 → 分流 → 导出 LeRobot → 读回自检。
#
#   tools/flywheel_once.sh                          # 全默认
#   tools/flywheel_once.sh --task pick_cube
#   SKIP_EXPORT=1 tools/flywheel_once.sh            # 只质检不导出
#
# 采集不在这个脚本里（要人戴手套）：先跑 record_episode.py 攒 episode，再跑本脚本。
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-./venv312/bin/python}
EPISODES=${EPISODES:-data/episodes}
CLEAN=${CLEAN:-data/clean}
REJECTED=${REJECTED:-data/rejected}
DATASET=${DATASET:-data/datasets/export}
REPO_ID=${REPO_ID:-local/wuji_glove_teleop}

TASK_ARG=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK_ARG=(--task "$2"); shift 2 ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -d "$EPISODES" ]]; then
  echo "没有 $EPISODES —— 先用 record_episode.py 采集" >&2
  exit 1
fi

echo "==> 1/3 质检 + 分流"
rm -rf "$CLEAN" "$REJECTED"
"$PY" tools/qc_episode.py "$EPISODES" --route link \
      --clean-dir "$CLEAN" --rejected-dir "$REJECTED"

# 注意：clean 目录可能压根没被创建（一条都没过），find 会失败，
# 在 set -e + pipefail 下会把脚本静默打断，所以先判目录存在。
N_CLEAN=0
if [[ -d "$CLEAN" ]]; then
  N_CLEAN=$(find "$CLEAN" -maxdepth 1 -mindepth 1 | wc -l)
fi
if [[ "$N_CLEAN" -eq 0 ]]; then
  echo; echo "没有通过质检的 episode，飞轮停在这一步。"
  echo "看 data/rejected/*/quality.json 里的 flags 定位原因，然后补采。"
  exit 0
fi

if [[ -n "${SKIP_EXPORT:-}" ]]; then
  echo; echo "SKIP_EXPORT=1，到此为止（clean=$N_CLEAN）"
  exit 0
fi

echo; echo "==> 2/3 导出 LeRobot 数据集"
"$PY" tools/export_dataset.py --input "$CLEAN" --out "$DATASET" \
      --repo-id "$REPO_ID" --overwrite "${TASK_ARG[@]}"

echo; echo "==> 3/3 读回自检"
"$PY" tools/export_dataset.py --verify "$DATASET" --repo-id "$REPO_ID"

echo
echo "飞轮跑完一圈：clean=$N_CLEAN → $DATASET"
echo "下一步（可选）训个 BC baseline 验证数据可训："
echo "  $PY tools/train_bc.py --dataset $DATASET --repo-id $REPO_ID --out data/models/bc.pt"
