import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import sys

class GlobalLLM:
    _tokenizer = None
    _model = None

    @staticmethod
    def load(model_path: str, dtype: torch.dtype = torch.float16, device: Optional[str] = None):
        if GlobalLLM._model is None:
            print(f"[GlobalLLM] Loading model from: {model_path} ...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                if device is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"

                model = model.to(device).eval()
                if device == "cuda":
                    model = model.half()

                GlobalLLM._tokenizer = tokenizer
                GlobalLLM._model = model
                print("[GlobalLLM] Loaded successfully.")
            except Exception as e:
                print(f"[GlobalLLM] Load failed: {e}")
                raise e

    @staticmethod
    def get(model_path: str = "/share/home/202230275320/workspace/medical_coding/model/glm-4-9B"):
        if GlobalLLM._model is None:
            GlobalLLM.load(model_path)
        return {
            "model": GlobalLLM._model,
            "tokenizer": GlobalLLM._tokenizer
        }

# ============================================================
# ⚠️ 调试核心区域 ⚠️
# ============================================================
def glm_chat(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """
    带深度调试打印的 chat 函数
    """
    print(f"\n[DEBUG] glm_chat called. Prompt len: {len(prompt)}")

    # 1) 优先尝试 .chat() 方法
    if hasattr(model, "chat"):
        print("[DEBUG] Model has .chat() method. Attempting call...")
        try:
            # ⭐ 关键点：这里 absolutely 没有解包，只用一个变量 raw_response 接收
            raw_response = model.chat(
                tokenizer,
                prompt,
                history=[],
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            
            # ⭐ 打印出它到底返回了什么！
            print(f"[DEBUG] model.chat returned type: {type(raw_response)}")
            if isinstance(raw_response, (list, tuple)):
                print(f"[DEBUG] Return length: {len(raw_response)}")
                print(f"[DEBUG] First element type: {type(raw_response[0])}")
            
            # 安全提取
            if isinstance(raw_response, (tuple, list)):
                if len(raw_response) > 0:
                    final_res = raw_response[0]
                else:
                    print("[DEBUG] Return list is empty!")
                    final_res = ""
            else:
                final_res = raw_response

            return str(final_res).strip()

        except ValueError as ve:
            # 专门捕获解包错误 (虽然这里没有解包，但以防万一)
            print(f"❌ [CRITICAL DEBUG] ValueError inside glm_chat: {ve}")
            import traceback
            traceback.print_exc()
            return ""
        except Exception as e:
            print(f"⚠️ [DEBUG] model.chat failed with: {e}. Falling back to generate.")
            # 继续往下走 fallback
            pass

    # 2) Fallback: Generate
    print("[DEBUG] Entering generate fallback...")
    try:
        device = next(model.parameters()).device
        # 显式添加 attention_mask 防止警告
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(device)
        
        prompt_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = outputs[0][prompt_length:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"[DEBUG] Generate success. Length: {len(text)}")
        return text.strip()
        
    except Exception as e:
        print(f"❌ [CRITICAL] Generate failed: {e}")
        import traceback
        traceback.print_exc()
        return ""