"""Kimi 模型兼容辅助：K2.5 已下线，统一迁移到 K3。"""

KIMI_K3_MODEL = "kimi-k3"
KIMI_K25_RETIRED_MODEL = "kimi-k2.5"
# K3 默认 reasoning_effort=max 偏慢且输出贵，实盘决策用 high 平衡质量与耗时
KIMI_K3_REASONING_EFFORT = "high"


def is_kimi_k3_model(model: str) -> bool:
    return "kimi-k3" in str(model or "").lower()


def migrate_retired_kimi_config(config: dict) -> dict:
    """把已下线的 kimi-k2.5 配置迁移到 kimi-k3。"""
    if not isinstance(config, dict):
        return config
    model_configs = dict(config.get("model_configs") or {})
    if KIMI_K25_RETIRED_MODEL in model_configs:
        if KIMI_K3_MODEL not in model_configs:
            model_configs[KIMI_K3_MODEL] = model_configs.pop(KIMI_K25_RETIRED_MODEL)
        else:
            model_configs.pop(KIMI_K25_RETIRED_MODEL, None)
        config["model_configs"] = model_configs
    if str(config.get("selected_model") or "") == KIMI_K25_RETIRED_MODEL:
        config["selected_model"] = KIMI_K3_MODEL
    return config
