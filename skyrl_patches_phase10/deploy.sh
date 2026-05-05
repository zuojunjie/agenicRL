#!/usr/bin/env bash
# Deploy Phase 10 patches to remote SkyRL instance.
#
# Usage:
#   bash deploy.sh <ssh_host> <ssh_port>
#   e.g. bash deploy.sh autodl-pro6k-2gpu 20683

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <ssh_host> <ssh_port>"
    exit 1
fi

HOST=$1
PORT=$2
LOCAL_DIR=$(cd "$(dirname "$0")" && pwd)
SKYRL_GYM=/root/autodl-tmp/external/SkyRL/skyrl-gym/skyrl_gym
SCRIPTS_DIR=/root/autodl-tmp/agenicRL/scripts
DATA_DIR=/root/autodl-tmp/data/math

echo "=== 1. 创建远端目录 ==="
ssh -p $PORT root@$HOST "mkdir -p $SKYRL_GYM/envs/math_python $SKYRL_GYM/tools $DATA_DIR"

echo "=== 2. 复制代码 ==="
# Python sandbox tool
scp -P $PORT $LOCAL_DIR/python_sandbox.py root@$HOST:$SKYRL_GYM/tools/python_sandbox.py
# math_python env
scp -P $PORT $LOCAL_DIR/math_python/__init__.py root@$HOST:$SKYRL_GYM/envs/math_python/__init__.py
scp -P $PORT $LOCAL_DIR/math_python/env.py root@$HOST:$SKYRL_GYM/envs/math_python/env.py
scp -P $PORT $LOCAL_DIR/math_python/utils.py root@$HOST:$SKYRL_GYM/envs/math_python/utils.py
# Data prep + launcher
scp -P $PORT $LOCAL_DIR/prep_data.py root@$HOST:/root/autodl-tmp/agenicRL/scripts/prep_data.py
scp -P $PORT $LOCAL_DIR/../scripts/skyrl_phase10_pro6k_2gpu_math_python.sh root@$HOST:$SCRIPTS_DIR/

echo "=== 3. 注册 env (修改 envs/__init__.py) ==="
ssh -p $PORT root@$HOST "
INIT_FILE=$SKYRL_GYM/envs/__init__.py
if ! grep -q 'math_python' \$INIT_FILE; then
    cp \$INIT_FILE \$INIT_FILE.bak
    cat >> \$INIT_FILE << 'EOF'

# Phase 10: MATH+Python env
try:
    from skyrl_gym.envs.math_python import MathPythonEnv, MathPythonEnvConfig
    register_env('math_python', MathPythonEnv, MathPythonEnvConfig)
except ImportError as e:
    pass  # math_python optional
EOF
    echo '已添加 math_python 到 envs/__init__.py'
else
    echo 'math_python 已注册'
fi
"

echo "=== 4. 准备数据 (GSM8K + MATH parquet) ==="
ssh -p $PORT root@$HOST "
source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1
cd $SCRIPTS_DIR
python3 prep_data.py --output-dir $DATA_DIR
ls -la $DATA_DIR
"

echo "=== 5. 自检 (sandbox + reward fn) ==="
ssh -p $PORT root@$HOST "
source /root/miniconda3/etc/profile.d/conda.sh
conda activate searchr1
cd $SKYRL_GYM
python3 tools/python_sandbox.py
python3 envs/math_python/utils.py
"

echo "=== 6. chmod launcher ==="
ssh -p $PORT root@$HOST "chmod +x $SCRIPTS_DIR/skyrl_phase10_pro6k_2gpu_math_python.sh"

echo ""
echo "✅ Phase 10 部署完成. 启动:"
echo "   ssh -p $PORT root@$HOST 'nohup bash -c \"export WANDB_API_KEY=\\\$(awk \\\"/api.wandb.ai/{getline; getline; print \\\\\\\\\\\$2}\\\" /root/.netrc); bash $SCRIPTS_DIR/skyrl_phase10_pro6k_2gpu_math_python.sh\" > /tmp/phase10-launch.out 2>&1 < /dev/null &'"
