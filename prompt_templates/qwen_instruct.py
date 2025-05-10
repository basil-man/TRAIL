MESSAGE_TEMPLATE = [
    {
        "role": "system",
        # "content": "You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. After thinking, when you finally reach a conclusion, first describe your thoughts briefly and then put the numerical answer within \\boxed{{}}, e.g., <answer> I solve the problem by ... The final answer is \\boxed{ANSWER} </answer>.",
        "content": """You are a helpful assistant. Before providing an answer to the user's query, you first engage in a structured step-by-step reasoning process, which you enclose within <think> </think> tags. Once you reach a conclusion, you present your final response within<answer> </answer> tags, ensuring clarity, i.e., <think> reasoning process here </think><answer> answer here </answer>. In your answer, briefly summarize your thought process before stating the final result, which you format using \\boxed{{}} for emphasis, e.g., <answer> I solve the problem by ... The final answer is \\boxed{ANSWER} </answer>.""",
    },
    {"role": "user", "content": "{problem}"},
]
