# This file defines functions for code correctness evaluation, which is used in RL training.

import multiprocessing
from verl.evaluation.code_eval import check_correctness, sanitize_code


# TODO (jiabao): test code function
def extract_code(solution_str):
    # assume the code is wrapped in ```python```
    return sanitize_code(solution_str)


def compute_score(
    solution_str, ground_truth, results_cache, lock, format_score=0.1
) -> float:
    # For coding, the ground_truth is actually the test cases
    code_str = extract_code(solution_str)

    if code_str is None or code_str == "":
        with lock:
            results_cache[(solution_str, ground_truth)] = 0
        return 0

    with lock:
        if (solution_str, ground_truth) in results_cache:
            return results_cache[(solution_str, ground_truth)]

    out_queue = multiprocessing.Queue()
    process_args = (code_str, ground_truth, out_queue)
    process = multiprocessing.Process(target=check_correctness, args=process_args)
    process.start()

    process.join(30)
    if process.is_alive():
        process.terminate()
        process.join()
        score = format_score
    else:
        try:
            acc = out_queue.get(timeout=1)
            score = 1 if acc else format_score
        except Exception as _:
            score = format_score
    with lock:
        results_cache[(solution_str, ground_truth)] = score
    return score, code_str
