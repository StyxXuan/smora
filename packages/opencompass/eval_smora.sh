#!/bin/bash

for i in {1..10}
do
    echo "开始运行第 $i 次"
    opencompass --models hf_llama3_8b_instruct_smora --datasets ds1000_service_eval_gen_cbc84f
    
    if [ $? -eq 0 ]; then
        echo "第 $i 次运行成功"
    else
        echo "第 $i 次运行失败" >&2
        exit 1
    fi
    
    echo "第 $i 次运行完成"
    echo "-----------------------------------"
done

