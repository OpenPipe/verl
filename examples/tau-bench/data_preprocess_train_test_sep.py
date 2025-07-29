"""
Preprocess the tau-bench dataset to parquet format
"""

import argparse
import asyncio
import os

from datasets import Dataset
from tqdm.asyncio import tqdm_asyncio

from verl.tau_bench.envs import get_env
from verl.utils.hdfs_io import copy, makedirs


async def create_datapoint(split, idx, env_name):
    env = get_env(
        env_name,
        user_strategy="llm",
        user_model="gpt-4.1",
        user_provider="openai",
        task_split=split,
        task_index=idx,
    )
    env_reset_res = await env.reset(task_index=idx)

    data = {
        "data_source": f"tau-bench-{split}",
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
        "ability": env_name,
        "reward_model": {"style": "rule", "ground_truth": "never gonna give you up, never gonna let you down"},
        "extra_info": {
            "split": split,
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
    return data


async def get_dataset(start_index, end_index, split, env_name):
    dataset = await tqdm_asyncio.gather(
        *[create_datapoint(split, idx, env_name) for idx in range(start_index, end_index)],
        desc=f"Creating {split} dataset",
    )

    return dataset


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/tau-bench")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_start_index", type=int, default=0)
    parser.add_argument("--train_end_index", type=int, default=32)
    parser.add_argument("--test_start_index", type=int, default=32)
    parser.add_argument("--test_end_index", type=int, default=115)
    parser.add_argument("--train_split_name", type=str, default="test")
    parser.add_argument("--test_split_name", type=str, default="test")
    parser.add_argument("--env_name", type=str, default="retail")

    args = parser.parse_args()

    train_dataset = await get_dataset(
        args.train_start_index, args.train_end_index, split=args.train_split_name, env_name=args.env_name
    )
    test_dataset = await get_dataset(
        args.test_start_index, args.test_end_index, split=args.test_split_name, env_name=args.env_name
    )

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
