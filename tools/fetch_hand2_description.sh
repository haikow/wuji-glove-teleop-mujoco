#!/usr/bin/env bash
# 拉二代手的 MJCF/mesh 到 wuji_hand_description2/（本仓库只 vendor 了一代）。
#
# 为什么需要：retarget 用 wujihand2 profile 时，输出落在**二代**关节限位内。实测
# 用一代 MJCF 去 clip，63.7% 的帧越界、最大 0.53 rad(30°) —— 回放和限位都是错的。
# 二代 beta1/beta2 的 20 个关节限位完全一致，这里取较新的 beta2。
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${DEST:-wuji_hand_description2}
SRC=hand2/hand2_beta2/body

if [[ -d "$DEST/mjcf" ]]; then
  echo "$DEST 已存在，跳过（要重拉先 rm -rf $DEST）"; exit 0
fi
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
echo "从 wuji-technology/wuji-description 拉 $SRC ..."
git clone --depth 1 --filter=blob:none --sparse \
    git@github.com:wuji-technology/wuji-description.git "$TMP/desc"
git -C "$TMP/desc" sparse-checkout set "$SRC"
mkdir -p "$DEST"
cp -r "$TMP/desc/$SRC/mjcf" "$TMP/desc/$SRC/meshes" "$DEST/"
[[ -f "$TMP/desc/LICENSE" ]] && cp "$TMP/desc/LICENSE" "$DEST/"
echo "完成：$DEST  ($(du -sh "$DEST" | cut -f1))"
echo "之后 --hand-model wuji_hand_2 会自动用它，不再回落一代 MJCF。"
