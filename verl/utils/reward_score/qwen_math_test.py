from contextlib import nullcontext

from absl.testing import parameterized

from verl.utils.reward_score.qwen_math import compute_score_r1

class SimpleTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            """<think>\nOkay, so I have this problem here: compute the sum from 1 to 100 where\n</think>\n<answer>-50</answer>""",
            -50,
        ),
        (
            """<think>\nOkay, so I have this problem here: \\boxed{25}\n</think>
            Therefore, the number of integer values of \( x \) is \(\boxed{25}\).<｜end▁of▁sentence｜>""",
            25,
        ),
    )
    def test_masks_constructed_from_segment_ids(
        self,
        solution_str,
        ground_truth,
    ):
        reward, answer = compute_score_r1(
            solution_str,
            ground_truth,
            results_cache={},
            lock=nullcontext(),
        )

        # assert reward == 1



