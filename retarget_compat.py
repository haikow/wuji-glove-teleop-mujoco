"""兼容 wuji-sdk 两种 retargeting 导入路径。

- **2026.7.21 / 2026.8.3 起**：retargeting 已原生化，`HandModel` / `RetargetSession`
  **顶层导出**（`from wuji_sdk import HandModel, RetargetSession`），且**不再需要 `[retarget]` extra**
  （无条件编入基础 wheel，`pip install "wuji-sdk"` 即可）。
- **≤ 2026.7.15**：闭源 retargeting 在 `wuji_sdk.retargeting` 子模块，
  需 `pip install "wuji-sdk[retarget]"`。

统一从本模块导入 `HandModel` / `RetargetSession`，两种 SDK 版本都能跑。
"""

try:  # 新（2026.7.21+ / 8.3，顶层）
    from wuji_sdk import HandModel, RetargetSession
except ImportError:  # 旧（≤7.15，子模块）
    from wuji_sdk.retargeting import HandModel, RetargetSession

__all__ = ["HandModel", "RetargetSession"]
