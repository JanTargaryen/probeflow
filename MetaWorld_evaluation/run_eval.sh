#!/bin/bash

set -euo pipefail

ROOT_DIR="/mnt/data_ssd/zhoufang/code/probeflow"
SERVER_DIR="$ROOT_DIR/Evo_1/scripts"
CLIENT_DIR="$ROOT_DIR/MetaWorld_evaluation"

GPU_ID=0
PORT=9010
CKPT_DIR="$ROOT_DIR/Evo_1/checkpoints/metaworld"
SOLVER="adaflow"
STEPS=""
SEEDS="42,123,2024,3407,10086"
EPISODES=1
EPISODE_HORIZON=200
TARGET_LEVEL="all"
EXP_NAME=""
SAVE_VIDEO=0
SAVE_IMAGE=0
ATTACH=1

usage() {
    cat <<'EOF'
Usage:
  bash run_eval.sh [options]

Options:
  --gpu ID                 GPU id for server/client. Default: 0
  --port PORT              Server port. Default: 9010
  --ckpt_dir PATH          Checkpoint dir to evaluate
  --solver NAME            adaflow | probeflow | euler | rk45 | dpm_multistep | heun
  --steps N                Required for fixed-step solvers like euler/heun/dpm_multistep/rk45
  --seeds CSV              Comma-separated seeds. Default: 42,123,2024,3407,10086
  --episodes N             Episodes per task. Default: 1
  --episode_horizon N      Max env steps per episode. Default: 200
  --target_level LEVEL     all | easy | medium | hard | very_hard
  --exp_name NAME          Explicit log-name prefix
  --save_video             Save rollout videos
  --save_image             Save inspect images
  --detach                 Start tmux session and do not attach
  -h, --help               Show this help

Examples:
  bash run_eval.sh \
    --gpu 1 \
    --port 9012 \
    --ckpt_dir /mnt/data_ssd/zhoufang/code/probeflow/Evo_1/checkpoints/adaflow_metaworld_stage2_live/step_best \
    --solver adaflow \
    --exp_name adaflow_mt50_5seed

  bash run_eval.sh \
    --gpu 2 \
    --port 9013 \
    --ckpt_dir /mnt/data_ssd/zhoufang/code/probeflow/Evo_1/checkpoints/metaworld \
    --solver probeflow \
    --exp_name probeflow_mt50_5seed
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --ckpt_dir)
            CKPT_DIR="$2"
            shift 2
            ;;
        --solver)
            SOLVER="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --episode_horizon)
            EPISODE_HORIZON="$2"
            shift 2
            ;;
        --target_level)
            TARGET_LEVEL="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --save_video)
            SAVE_VIDEO=1
            shift
            ;;
        --save_image)
            SAVE_IMAGE=1
            shift
            ;;
        --detach)
            ATTACH=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ ! -d "$CKPT_DIR" ]]; then
    echo "[ERROR] Checkpoint dir not found: $CKPT_DIR"
    exit 1
fi

if [[ -z "$EXP_NAME" ]]; then
    EXP_NAME="$(basename "$CKPT_DIR")_${SOLVER}"
fi

if [[ "$SOLVER" != "adaflow" && "$SOLVER" != "probeflow" && -z "$STEPS" ]]; then
    echo "[ERROR] --steps is required for fixed-step solver '$SOLVER'"
    exit 1
fi

SESSION_SAFE_NAME="$(echo "$EXP_NAME" | tr '/: ' '___')"
SESSION="eval_${SESSION_SAFE_NAME}_${PORT}"

CLIENT_CMD=(
    python mt50_evo1_client_prompt.py
    --port "$PORT"
    --ckpt_dir "$CKPT_DIR"
    --exp_name "$EXP_NAME"
    --solver "$SOLVER"
    --seeds "$SEEDS"
    --episodes "$EPISODES"
    --episode_horizon "$EPISODE_HORIZON"
    --target_level "$TARGET_LEVEL"
)

if [[ -n "$STEPS" ]]; then
    CLIENT_CMD+=(--steps "$STEPS")
fi

if [[ "$SAVE_VIDEO" -eq 1 ]]; then
    CLIENT_CMD+=(--save_video)
fi

if [[ "$SAVE_IMAGE" -eq 1 ]]; then
    CLIENT_CMD+=(--save_image)
fi

echo "=================================================="
echo "Launching MetaWorld Evaluation"
echo "Session      : $SESSION"
echo "GPU          : $GPU_ID"
echo "Port         : $PORT"
echo "Checkpoint   : $CKPT_DIR"
echo "Solver       : $SOLVER"
echo "Steps        : ${STEPS:-adaptive}"
echo "Seeds        : $SEEDS"
echo "Episodes     : $EPISODES"
echo "Target Level : $TARGET_LEVEL"
echo "Exp Name     : $EXP_NAME"
echo "=================================================="

echo "[INFO] Cleaning up port $PORT..."
PORT_PIDS="$(lsof -ti:"$PORT" 2>/dev/null || true)"
if [[ -n "$PORT_PIDS" ]]; then
    echo "$PORT_PIDS" | xargs -r kill -9
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[INFO] Killing previous tmux session '$SESSION'..."
    tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION"
tmux split-window -h -t "$SESSION"

tmux send-keys -t "$SESSION":0.0 "source ~/.bashrc" C-m
tmux send-keys -t "$SESSION":0.0 "export TRANSFORMERS_OFFLINE=1" C-m
tmux send-keys -t "$SESSION":0.0 "export HF_HUB_OFFLINE=1" C-m
tmux send-keys -t "$SESSION":0.0 "export SWANLAB_MODE=disabled" C-m
tmux send-keys -t "$SESSION":0.0 "conda activate evo" C-m
tmux send-keys -t "$SESSION":0.0 "export CUDA_VISIBLE_DEVICES=$GPU_ID" C-m
tmux send-keys -t "$SESSION":0.0 "cu124" C-m
tmux send-keys -t "$SESSION":0.0 "cd $SERVER_DIR" C-m
tmux send-keys -t "$SESSION":0.0 "python Evo1_server.py --port $PORT --ckpt_dir \"$CKPT_DIR\"" C-m

tmux send-keys -t "$SESSION":0.1 "source ~/.bashrc" C-m
tmux send-keys -t "$SESSION":0.1 "conda activate metaworld" C-m
tmux send-keys -t "$SESSION":0.1 "export CUDA_VISIBLE_DEVICES=$GPU_ID" C-m
tmux send-keys -t "$SESSION":0.1 "cd $CLIENT_DIR" C-m
tmux send-keys -t "$SESSION":0.1 "echo '[INFO] Waiting 90s for server warmup...'" C-m
tmux send-keys -t "$SESSION":0.1 "sleep 90" C-m

CLIENT_CMD_STR="$(printf '%q ' "${CLIENT_CMD[@]}")"
tmux send-keys -t "$SESSION":0.1 "$CLIENT_CMD_STR; EXIT_CODE=\$?; if [ \$EXIT_CODE -eq 0 ]; then echo '[INFO] Evaluation finished successfully. Auto-destroying tmux session in 5s...'; sleep 5; tmux kill-session -t \"$SESSION\"; else echo '[ERROR] Evaluation failed with exit code '\$EXIT_CODE'. Session kept for debugging.'; exec bash; fi" C-m

if [[ "$ATTACH" -eq 1 ]]; then
    echo "[SUCCESS] Attaching to tmux session: $SESSION"
    tmux attach -t "$SESSION"
else
    echo "[SUCCESS] Started tmux session: $SESSION"
    echo "Inspect with: tmux attach -t $SESSION"
fi
