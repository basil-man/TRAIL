# We follow the implementation in BigcodeBench&apps for coding evaluation implementation.
# The BigcodeBench project employs Apache-2.0 License. We refer to the license file in original repository (https://github.com/bigcode-project/bigcodebench) for more details.
# The apps project employs MIT License. We refer to the license file in original repository (https://github.com/hendrycks/apps) for more details.

import json
from typing import Any, Dict, List, Tuple

from .apps.apps_utils import run_test as apps_run_test
from .bigcodebench.code_utils import untrusted_check as bigcodebench_untrusted_check
from .bigcodebench.code_utils import PASS as bigcodebench_PASS
from .bigcodebench.sanitize import sanitize as sanitize_code

# 1st item: the status
# 2nd item (optional): the detailed pass/fail boolean for each input
Result = Tuple[str, List[bool]]


# TODO jiabao: add test function for this part
def check_correctness(
    solution: str,
    testcase: str,
    entry_point: str = None,
    max_as_limit: float = 20 * 1024,
    max_data_limit: float = 20 * 1024,
    max_stack_limit: float = 10,
    min_time_limit: float = 0.1,
    gt_time_limit: float = 2.0,
):
    """Following the BigcodeBench implementation, we check the correctness of the solution.

    Args:
        solution (str):
            model completed solution code, e.g., "def task_func(x): return x", where task_func is the default function name. Remember to check whether the model generated solution contains this function.
        testcase:
            a string of code containing all test cases in unittest format. Example wrapped in following code block (excluding the code block wrapper):
                ```python
                import unittest
                from unittest.mock import patch
                from random import seed, shuffle
                import itertools
                class TestCases(unittest.TestCase):
                    def test_default_numbers(self):
                        # Test with default number range (1 to 10) to check that the result is a positive float.
                        result = task_func()
                        self.assertIsInstance(result, float)
                        self.assertGreater(result, 0)
                ```
        max_as_limit (float): limiting the address space when executing
        max_data_limit (float): limiting data space when executing
        max_stack_limit (float): limiting stack space when executing
        min_time_limit (float, optional): time limit for runnign. Defaults to 0.1.
        gt_time_limit (float, optional): timie limit in running. Defaults to 2.0.

    Returns:
        Dict[str, Result]: _description_
    """

    try:
        testcase = json.loads(testcase)
        # In apps implementation, testcase is a json string containing input/output
    except Exception as _:
        # In BigcodeBench implementation, testcase is a string of python code
        testcase = testcase

    if isinstance(testcase, dict):
        # apps code
        result = apps_run_test(
            solution,
            test=testcase,
        )
        result = all(result)
    else:
        # BigcodeBench code
        result, _ = bigcodebench_untrusted_check(
            solution,
            test_code=testcase,
            entry_point=entry_point,
            max_as_limit=max_as_limit,
            max_data_limit=max_data_limit,
            max_stack_limit=max_as_limit,
            min_time_limit=min_time_limit,
            gt_time_limit=gt_time_limit,
        )
        result = result == bigcodebench_PASS

    return result


__ALL__ = ["check_correctness", "sanitize_code"]
