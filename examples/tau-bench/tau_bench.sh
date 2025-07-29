# run on 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/tau-bench/config"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='tau_bench_config' \
    data.train_files=$HOME/data/tau-bench/train.parquet \
    data.val_files=$HOME/data/tau-bench/test.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/tau-bench/config/tool_config/tau_bench_tool_config.yaml" $@