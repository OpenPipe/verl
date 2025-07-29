# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Compute score for tau-bench by extracting reward from the rollout pipeline.

    The tau-bench reward is passed through the rollout pipeline in the reward_scores.
    Instead of re-evaluating the solution, we extract the pre-computed reward.

    Note: This function is designed to work with a custom reward manager that
    passes reward_scores in extra_info, or it can be used by patching the
    naive reward manager to transfer reward_scores to extra_info.

    Args:
        data_source: The data source identifier (not used for tau-bench)
        solution_str: The solution string from the model (not used for tau-bench)
        ground_truth: The ground truth answer (not used for tau-bench)
        extra_info: Extra information that should contain reward_scores
        **kwargs: Additional arguments

    Returns:
        float: The tau-bench reward score
    """
    # print("inside compute_score")
    if extra_info is None:
        return 0.0

    reward = extra_info.get("reward", 0.0)

    return reward
