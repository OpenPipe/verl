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

import asyncio
import copy
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from verl.workers.rollout.schemas import Message


class RolloutScore(BaseModel):
    """
    Model representing the score and explanation for a single rollout.
    """

    rollout_index: int = Field(description="Index of the rollout being scored")
    explanation: str = Field(description="Detailed explanation of what the rollout did well and what it did poorly")
    score: float = Field(description="Numerical score between 0 and 1 indicating rollout quality")


class RolloutScores(BaseModel):
    """
    Model representing scores for a group of rollouts.
    """

    rollout_scores: List[RolloutScore] = Field(
        description="List of RolloutScore objects containing indices, explanations, and scores for each rollout"
    )


GENERAL_RM_PROMPT = """All of the rollouts below have been given the same task. Your job is to consider each of them and give them a score between 0 and 1. Take into consideration your best judgement of the task's intended outcome. 

Grading standards:
- A rollout that achieves its goal should always get a significantly higher score than a rollout that does not achieve its goal.
- A rollout that achieves its goal more efficiently (eg. by avoiding unproductive detours) should get a higher score than a rollout that achieves its goal less efficiently.
- If one rollout is only slightly better than another, the difference in scores should be small. If it is significantly better, the difference in scores should be large.
- You may give some partial credit for a rollout that makes progress towards the task but does not complete it.

For each rollout, you need to output the rollout index, the explanation of the score, and the score. The score should be between 0 and 1.

"""


def create_and_split_messages(messages: List[Message]) -> Tuple[str, List[dict[str, Any]]]:
    """
    Create a system message from the first message in the group and return the remaining messages.

    Note: This is a stub function. Implementation will be managed externally.
    """
    messages_copy = copy.deepcopy(messages)
    system_message = ""
    if messages_copy[0].role == "system":
        system_message = messages_copy[0].content
    else:
        print(f"WARNING: first message is not a system message: {messages_copy[0]}")

    remaining_messages = [msg.model_dump() for msg in messages_copy[1:]]
    remaining_messages = [{k: v for k, v in msg.items() if v is not None} for msg in remaining_messages]
    return system_message, remaining_messages


def add_messages_to_prompt(user_prompt: str, messages: List[Dict[str, Any]]) -> str:
    """
    Add messages to the prompt in a formatted way.
    """
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if tool_calls and len(tool_calls) > 0:
            tool_call_str = ""
            for tool_call in tool_calls:
                tool_call_str += f"TOOL CALL: {tool_call['function']['name']}: {tool_call['function']['arguments']}\n"
            user_prompt += f"{role.upper()}: {tool_call_str}\n"
        else:
            user_prompt += f"{role.upper()}: {content}\n"
    return user_prompt


def count_assistant_messages(messages: List[Message]) -> int:
    """
    Count the number of assistant messages in a conversation.
    """
    return sum(1 for msg in messages if msg.role == "assistant")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=15),
    reraise=True,
)
async def create_openai_response(prompt: str, judge_model: str = "o3") -> RolloutScores:
    """
    Create an OpenAI response for scoring rollouts.
    """

    # with open(f"prompt_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}.txt", "w") as f:
    #     f.write(prompt)

    async with AsyncOpenAI() as client:
        try:
            response = await client.beta.chat.completions.parse(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                response_format=RolloutScores,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            raise e
        return response.choices[0].message.parsed


async def score_task_group(group_items: List[Tuple[int, Dict[str, Any]]]) -> List[Tuple[int, float]]:
    """
    Score a group of items with the same task index.
    Returns list of (original_index, score) tuples.
    """
    user_prompt = GENERAL_RM_PROMPT

    # Use the first item's messages to extract system prompt
    first_messages = group_items[0][1]["messages"]
    system_message, remaining_messages = create_and_split_messages(first_messages)

    user_prompt += f"Here is the system prompt that was provided at the beginning of each of the rollouts:\n--- START OF SYSTEM PROMPT ---\n{system_message}\n--- END OF SYSTEM PROMPT ---\n\n Here are the rollouts to evaluate:"

    # Add each rollout to the prompt
    for idx, (orig_idx, extra_info) in enumerate(group_items):
        user_prompt += f"\n\n--- ROLLOUT {idx} ---\n"
        _, messages = create_and_split_messages(extra_info["messages"])
        user_prompt = add_messages_to_prompt(user_prompt, messages)

    # Get scores from LLM
    try:
        response = await create_openai_response(user_prompt)
    except Exception as e:
        print(f"ERROR: could not compute scores for group: {e}")
        response = RolloutScores(
            rollout_scores=[
                RolloutScore(
                    rollout_index=idx,
                    explanation="error computing scores",
                    score=-1.0,
                )
                for idx in range(len(group_items))
            ]
        )

    # Map scores back to original indices
    results = []
    for idx, (orig_idx, extra_info) in enumerate(group_items):
        score = response.rollout_scores[idx].score
        results.append((orig_idx, score))

    return results


def compute_score(data_sources, solution_strs, ground_truths, extra_infos=None, **kwargs):
    """
    Compute score for tau-bench using LLM-based evaluation.

    Groups samples by task index and makes LLM calls to score each group.
    Adjusts rewards based on existing reward values and number of assistant messages.

    Args:
        data_sources: The data source identifiers
        solution_strs: The solution strings from the model
        ground_truths: The ground truth answers
        extra_infos: List of dicts containing task_idx, messages, and reward
        **kwargs: Additional arguments

    Returns:
        List[float]: The tau-bench reward scores in the same order as inputs
    """
    if extra_infos is None:
        raise ValueError("extra_infos is required for tau-bench scoring")

    # Group items by task index
    task_groups = defaultdict(list)
    for idx, extra_info in enumerate(extra_infos):
        task_idx = extra_info.get("task_idx")
        if task_idx is not None:
            task_groups[task_idx].append((idx, extra_info))

    # Score each group asynchronously
    async def score_all_groups():
        tasks = []
        for task_idx, group_items in task_groups.items():
            tasks.append(score_task_group(group_items))
        return await asyncio.gather(*tasks)

    # Run the async scoring
    all_scores = asyncio.run(score_all_groups())

    # Create result array with same length as inputs
    final_scores = [{} for _ in range(len(extra_infos))]

    # Process scores and apply reward adjustments
    for group_scores in all_scores:
        for orig_idx, llm_score in group_scores:
            existing_reward = extra_infos[orig_idx]["reward"]
            messages = extra_infos[orig_idx]["messages"]
            num_assistant_messages = count_assistant_messages(messages)

            final_scores[orig_idx]["outcome_correct"] = existing_reward
            final_scores[orig_idx]["llm_score"] = llm_score
            final_scores[orig_idx]["num_assistant_turns"] = num_assistant_messages

            if llm_score == -1.0:
                final_scores[orig_idx]["score"] = existing_reward
            elif existing_reward == 0:
                # If existing reward is 0, use the LLM score
                final_scores[orig_idx]["score"] = llm_score
            elif existing_reward == 1:
                # If existing reward is 1, keep it as 1 and add turn bonus
                turn_bonus = (30 - num_assistant_messages) / 30
                final_scores[orig_idx]["score"] = 1.0 + turn_bonus
            else:
                print(f"WARNING: existing_reward is {existing_reward} for orig_idx {orig_idx}")
                # For other values, use the existing reward
                final_scores[orig_idx]["score"] = existing_reward

    return final_scores
