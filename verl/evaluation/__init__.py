from verl.utils.registry_utils import Registry

from .math_utils import (
    qwen_math_equal,
    deepseek_math_equal,
    VerLMathEqual,
    pass_at_k,
    qwen_instruct_math_equal,
)
from .code_eval import check_correctness

REWARD_REGISTRY = Registry()

REWARD_REGISTRY.register("qwen_math", qwen_math_equal)
REWARD_REGISTRY.register("qwen_math_instruct", qwen_instruct_math_equal)
REWARD_REGISTRY.register("deepseek_math", deepseek_math_equal)
REWARD_REGISTRY.register("verl_math", VerLMathEqual)
REWARD_REGISTRY.register("code_all", check_correctness)


__ALL__ = ["REWARD_REGISTRY", "pass_at_k"]
