"""
Preprocess the tau-bench dataset to parquet format
"""

import argparse
import asyncio
import os

from datasets import Dataset

from verl.tau_bench.envs import get_env
from verl.utils.hdfs_io import copy, makedirs


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/tau_bench_verl_retail")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    end_index = 92
    test_start_index = 32

    train_dataset = []
    test_dataset = []

    for idx in range(end_index):
        env = get_env(
            "retail",
            user_strategy="llm",
            user_model="gpt-4.1",
            user_provider="openai",
            task_split="test",
            task_index=idx,
        )
        env_reset_res = await env.reset(task_index=idx)

        data = {
            "data_source": "tau-bench",
            "prompt": [
                {
                    "role": "system",
                    "content": env.wiki,
                },
                {
                    "role": "user",
                    "content": env_reset_res.observation,
                },
            ],
            "ability": "retail",
            "reward_model": {"style": "rule", "ground_truth": "never gonna give you up, never gonna let you down"},
            "extra_info": {
                "split": "train" if idx < test_start_index else "test",
                "index": idx,
                "instruction": env.task.instruction,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "sample_tau_tool_call": {
                        "create_kwargs": {"test_tool_call_key": "test_tool_call_value"},
                        # "execute_kwargs": {},
                        # "calc_reward_kwargs": {},
                        # "release_kwargs": {},
                    },
                },
            },
        }
        if idx < test_start_index:
            train_dataset.append(data)
        else:
            test_dataset.append(data)

    train_dataset = Dataset.from_list(train_dataset)
    test_dataset = Dataset.from_list(test_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)


if __name__ == "__main__":
    asyncio.run(main())
