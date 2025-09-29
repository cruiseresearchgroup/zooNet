#!/bin/bash

# Set environment variables
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 初始化 conda（解决 tmux 中的问题）
CONDA_PATH="/mnt/data728/duyin/anaconda3/bin/conda"
eval "$("$CONDA_PATH" shell.bash hook)"

# Initialize conda for bash script
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate virtual environment (if using)
conda activate hetero

echo "=========================================="
echo "运行改进的感染预测模型训练"
echo "针对损失函数不匹配和数据不平衡问题的优化"
echo "=========================================="

# 实验配置选择
EXPERIMENT=${1:-"infection_weighted"}  # 默认使用感染加权损失


case $EXPERIMENT in
    "baseline")
        echo "运行基线实验 (MSE损失)"
        python spactral_simple_main.py \
            --dataset alabama \
            --batch_size 16 \
            --window_size 4 \
            --pred_horizon 4 \
            --hidden_dim 64 \
            --lr 0.001 \
            --weight_decay 0.0001 \
            --dropout 0.1 \
            --epochs 100 \
            --model_dir './saved_results/state/alabama/baseline_mse' \
            --data_dir '/mnt/data728/datasets/state_graphs/alabama' \
            --spectral_gamma 0.2 \
            --use_cuda 'cuda' \
            --device '7' \
            --loss_type 'mse' \
            --use_post_processing True \
            --detection_threshold 0.3
        ;;
    
    "infection_weighted")
        echo "运行感染加权MSE损失实验 (推荐)"
        python spactral_simple_main.py \
            --dataset alabama \
            --batch_size 16 \
            --window_size 4 \
            --pred_horizon 4 \
            --hidden_dim 64 \
            --lr 0.001 \
            --weight_decay 0.0001 \
            --dropout 0.3 \
            --epochs 100 \
            --model_dir './saved_results/state/alabama/infection_weighted' \
            --data_dir '/mnt/data728/datasets/state_graphs/alabama' \
            --spectral_gamma 1 \
            --use_cuda 'cuda' \
            --device '7' \
            --loss_type 'infection_weighted' \
            --infection_zero_weight 1.0 \
            --infection_low_weight 8.0 \
            --infection_med_weight 15.0 \
            --infection_high_weight 25.0 \
            --infection_low_threshold 0.5 \
            --infection_med_threshold 5.0 \
            --infection_high_threshold 20.0 \
            --use_post_processing True \
            --detection_threshold 0.3 \
            --min_prediction 0.0
        ;;
    
    "focal")
        echo "运行改进Focal损失实验"
        python spactral_simple_main.py \
            --dataset alabama \
            --batch_size 16 \
            --window_size 4 \
            --pred_horizon 4 \
            --hidden_dim 64  \
            --lr 0.0008 \
            --weight_decay 0.0001 \
            --dropout 0.3 \
            --epochs 100 \
            --model_dir './saved_results/state/alabama/focal_loss' \
            --data_dir '/mnt/data728/datasets/state_graphs/alabama' \
            --spectral_gamma 0.1 \
            --use_cuda 'cuda' \
            --device '7' \
            --loss_type 'focal' \
            --focal_gamma 2.5 \
            --use_post_processing True \
            --detection_threshold 0.25
        ;;
    
    "hierarchical")
        echo "运行分层损失实验 (检测+回归)"
        python spactral_simple_main.py \
            --dataset alabama \
            --batch_size 16 \
            --window_size 4 \
            --pred_horizon 4 \
            --hidden_dim 64 \
            --lr 0.0005 \
            --weight_decay 0.0001 \
            --dropout 0.3 \
            --epochs 100 \
            --model_dir './saved_results/state/alabama/hierarchical' \
            --data_dir '/mnt/data728/datasets/state_graphs/alabama' \
            --spectral_gamma 0.1 \
            --use_cuda 'cuda' \
            --device '7' \
            --loss_type 'hierarchical' \
            --hierarchical_detection_weight 1.5 \
            --hierarchical_regression_weight 1.0 \
            --hierarchical_threshold 0.5 \
            --use_post_processing True \
            --detection_threshold 0.3
        ;;
    
    "combined")
        echo "运行组合损失实验 (MSE+Focal+Weighted)"
        python spactral_simple_main.py \
            --dataset alabama \
            --batch_size 16 \
            --window_size 4 \
            --pred_horizon 4 \
            --hidden_dim 64 \
            --lr 0.0008 \
            --weight_decay 0.0001 \
            --dropout 0.25 \
            --epochs 120 \
            --model_dir './saved_results/state/alabama/combined_loss' \
            --data_dir '/mnt/data728/datasets/state_graphs/alabama' \
            --spectral_gamma 0.1 \
            --use_cuda 'cuda' \
            --device '7' \
            --loss_type 'combined' \
            --focal_gamma 2.0 \
            --infection_zero_weight 1.0 \
            --infection_low_weight 6.0 \
            --infection_med_weight 12.0 \
            --infection_high_weight 20.0 \
            --use_post_processing True \
            --detection_threshold 0.3
        ;;
    
    "huber")
        echo "运行自适应Huber损失实验"
        python spactral_simple_main.py \
            --dataset alabama \
            --batch_size 16 \
            --window_size 4 \
            --pred_horizon 4 \
            --hidden_dim 64 \
            --lr 0.001 \
            --weight_decay 0.0001 \
            --dropout 0.3 \
            --epochs 100 \
            --model_dir './saved_results/state/alabama/huber_loss' \
            --data_dir '/mnt/data728/datasets/state_graphs/alabama' \
            --spectral_gamma 0.1 \
            --use_cuda 'cuda' \
            --device '7' \
            --loss_type 'huber' \
            --use_post_processing True \
            --detection_threshold 0.35
        ;;
    
    *)
        echo "未知实验类型: $EXPERIMENT"
        echo "可用选项: baseline, infection_weighted, focal, hierarchical, combined, huber"
        exit 1
        ;;
esac

echo "=========================================="
echo "训练完成!"
echo "检查结果目录: ./saved_results/avian/$EXPERIMENT"
echo "=========================================="

deactivate


