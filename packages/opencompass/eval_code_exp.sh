#!/bin/bash

# 定义包含训练脚本的文件夹路径
SCRIPT_DIR="."

# 定义要按顺序执行的脚本列表
scripts=(
    "eval_smora_filtered.sh"
    "eval_smora_greedy.sh"
    "eval_lora.sh"
)

# 遍历脚本列表并依次执行
for script in "${scripts[@]}"; do
    script_path="$SCRIPT_DIR/$script"
    
    # 检查文件是否存在并且可执行
    if [ -f "$script_path" ]; then
        echo "正在执行脚本: $script_path"
        # 执行脚本
        bash "$script_path"
        
        # 检查上一个命令的退出状态
        if [ $? -eq 0 ]; then
            echo "脚本 $script_path 执行成功"
        else
            echo "脚本 $script_path 执行失败"
            # 如果需要，可以在失败时退出循环
            # exit 1
        fi
    else
        echo "跳过不可执行的文件: $script_path"
    fi
done

echo "所有脚本执行完毕"