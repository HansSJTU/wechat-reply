"""API keys: environment variables first, then secrets/stt_keys.toml (mode 600, never commit)."""
import os
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS_FILE = os.path.join(ROOT, "secrets", "stt_keys.toml")

# name in stt_keys.toml -> environment variable
NAMES = {
    "openai_api_key": "OPENAI_API_KEY",
    "deepseek_api_key": "DEEPSEEK_API_KEY",
    "volc_app_id": "VOLC_APP_ID",
    "volc_access_token": "VOLC_ACCESS_TOKEN",
    "volc_api_key": "VOLC_API_KEY",
    "dashscope_api_key": "DASHSCOPE_API_KEY",
    "aliyun_workspace_id": "ALIYUN_WORKSPACE_ID",
    "tencent_secret_id": "TENCENTCLOUD_SECRET_ID",
    "tencent_secret_key": "TENCENTCLOUD_SECRET_KEY",
    "tencent_app_id": "TENCENTCLOUD_APP_ID",
    "xf_app_id": "XF_APP_ID",
    "xf_api_key": "XF_API_KEY",
    "xf_api_secret": "XF_API_SECRET",
    "siliconflow_api_key": "SILICONFLOW_API_KEY",
    "gemini_api_key": "GEMINI_API_KEY",
}

_file_cache = None


def get(name: str) -> str:
    global _file_cache
    v = os.environ.get(NAMES[name], "").strip()
    if v:
        return v
    if _file_cache is None:
        try:
            with open(KEYS_FILE, "rb") as f:
                _file_cache = tomllib.load(f)
        except FileNotFoundError:
            _file_cache = {}
    return str(_file_cache.get(name, "")).strip()
