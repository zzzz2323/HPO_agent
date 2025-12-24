import os
import yaml
from jinja2 import Template

class PromptManager:
    _instance = None
    
    def __init__(self, config_path="config/prompts.yaml"):
        # 自动寻找配置文件路径
        if not os.path.exists(config_path):
            # 尝试向上查找
            base_dir = os.path.dirname(os.path.dirname(__file__))
            config_path = os.path.join(base_dir, "config", "prompts.yaml")
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Prompt config not found at: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)
    
    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = PromptManager()
        return cls._instance

    def render(self, stage: str, template_name: str, **kwargs) -> str:
        """
        渲染 Prompt 模板
        Args:
            stage: 'preprocess', 'normalization', 'assertion', 'postcoordination'
            template_name: e.g., 'classification', 'binding'
            **kwargs: 模板变量
        Returns:
            渲染后的完整 Prompt 字符串
        """
        try:
            # 读取用户模板
            template_str = self.prompts[stage][template_name]["user_template"]
            
            # 读取可选的 system prompt (如果有些模型需要拼接)
            system_str = self.prompts[stage][template_name].get("system", "")
            
            # 使用 Jinja2 渲染
            template = Template(template_str)
            rendered_user = template.render(**kwargs)
            
            # 这里我们只返回 user 部分，因为 system prompt 通常由 Client 处理
            # 如果需要拼在一起，可以在这里拼
            # return f"{system_str}\n\n{rendered_user}" if system_str else rendered_user
            
            return rendered_user
            
        except KeyError:
            # 容错：如果找不到，返回一个简单的 fallback
            return f"[Error: Prompt template for {stage}.{template_name} not found]"
        except Exception as e:
            return f"[Error rendering prompt: {str(e)}]"