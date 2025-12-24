import os
import yaml
import logging
from typing import Optional, Tuple, Dict, Any

# 1. 导入您封装好的 Local 模块 (包含 GlobalLLM 和 glm_chat)
try:
    from core.global_llm import GlobalLLM, glm_chat
    LOCAL_LIB_AVAILABLE = True
except ImportError:
    try:
        from .global_llm import GlobalLLM, glm_chat
        LOCAL_LIB_AVAILABLE = True
    except ImportError:
        LOCAL_LIB_AVAILABLE = False
        pass

# 2. 导入 API 模块
try:
    from openai import OpenAI
    API_LIB_AVAILABLE = True
except ImportError:
    API_LIB_AVAILABLE = False

# 3. 加载配置
def _load_config():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "settings.yaml")
    
    if not os.path.exists(config_path):
        return {"pipeline_mode": "no_llm"}
        
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CONF = _load_config()
MODE = CONF.get("pipeline_mode", "no_llm")

class UnifiedLLMClient:
    _instance = None

    def __init__(self):
        self.mode = MODE
        self.client = None
        self.local_pack = None
        
        logging.info(f"Initializing UnifiedLLMClient in mode: {self.mode}")

        if self.mode == "local":
            if not LOCAL_LIB_AVAILABLE:
                raise RuntimeError("Local mode enabled but 'core.global_llm' not found.")
            self.local_pack = GlobalLLM.get()

        elif self.mode == "api":
            if not API_LIB_AVAILABLE:
                raise RuntimeError("API mode enabled but 'openai' not installed.")
            api_conf = CONF.get("api_config", {})
            self.client = OpenAI(
                base_url=api_conf.get("base_url"),
                api_key=api_conf.get("api_key")
            )
            self.model_name = api_conf.get("model_name", "gpt-3.5-turbo")

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = UnifiedLLMClient()
        return cls._instance

    def is_enabled(self) -> bool:
        return self.mode != "no_llm"

    def chat(self, prompt: str, system_prompt: str = None, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
        """
        统一接口：直接调用底层实现
        """
        if self.mode == "no_llm":
            return ""

        try:
            # === 本地模式：直接调用您写好的 glm_chat ===
            if self.mode == "local":
                model = self.local_pack["model"]
                tokenizer = self.local_pack["tokenizer"]
                
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                
                # ⭐⭐⭐ 重点检查这里！⭐⭐⭐
                # 必须是 return glm_chat(...) 
                # 绝对不能是 response, _ = glm_chat(...)
                # 绝对不能是 response, _ = model.chat(...)
                return glm_chat(model, tokenizer, full_prompt, max_new_tokens=max_new_tokens)

            # === API 模式 ===
            elif self.mode == "api":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content

        except Exception as e:
            # 打印详细错误栈，帮我们定位
            import traceback
            traceback.print_exc()
            logging.error(f"LLM Generation Error ({self.mode}): {e}")
            return ""
        
        return ""