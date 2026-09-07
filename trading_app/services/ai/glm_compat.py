"""智谱 GLM 模型兼容辅助。"""

GLM_FLASH_MODEL = "glm-5.3-flash"
GLM_DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
# 官方默认 reasoning_effort=max；实盘决策用 high 平衡质量与耗时
GLM_REASONING_EFFORT = "high"


def is_glm_model(model: str) -> bool:
    name = str(model or "").lower()
    return name.startswith("glm-") or "glm-" in name


def resolve_glm_base_url(base_url: str = "") -> str:
    return str(base_url or "").strip() or GLM_DEFAULT_BASE_URL


def build_glm_extra_body(model: str) -> dict:
    """GLM-5.3 / Flash 始终思考，用 extra_body 传递官方推荐参数。"""
    extra: dict = {}
    name = str(model or "").lower()
    if "glm-5.3" in name or "glm-5.2" in name:
        extra["reasoning_effort"] = GLM_REASONING_EFFORT
        extra["thinking"] = {"type": "enabled", "clear_thinking": False}
    return extra
