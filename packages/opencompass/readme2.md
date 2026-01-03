```bash
cd work/acl-2025-moe-lora/code-evaluator/
conda activate code_eval
python server.py --port 5000
```


```bash
cd work/acl-2025-moe-lora/opencompass/
opencompass --models hf_llama3_8b_instruct_smora --datasets ds1000_service_eval_gen_cbc84f
opencompass --models hf_llama3_8b_instruct_lora --datasets ds1000_service_eval_gen_cbc84f
```

opencompass --models hf_llama2_7b_lora hf_llama2_7b_smora --datasets humaneval_gen_8e312c

opencompass --models hf_llama2_7b_moe_soft hf_llama2_7b_moe_top1 hf_llama2_7b_moe_top2 hf_llama2_7b_hydralora hf_llama2_7b_moslora hf_llama2_7b_smear --datasets humaneval_repeat10_gen_8e312c gsm8k_gen



opencompass --models hf_llama2_7b_moe_soft hf_llama2_7b_hydralora --datasets mmlu_gen_4d595a humaneval_gen_8e312c


opencompass --models hf_llama2_7b_smora_wo_blc --datasets mmlu_gen_4d595a humaneval_gen_8e312c

opencompass --models hf_llama2_7b_smora_wo_blc --datasets humaneval_repeat10_gen_8e312c gsm8k_gen

opencompass --models hf_llama2_7b_smora_p8 --datasets mmlu_gen_4d595a humaneval_gen_8e312c

python run.py --models hf_llama2_7b_smora_p1 hf_llama2_7b_smora_p2 hf_llama2_7b_smora_p4 --datasets mmlu_gen_4d595a humaneval_gen_8e312c -r 20250329_092952



python run.py --models hf_llama2_7b_lora hf_llama2_7b_moe_soft hf_llama2_7b_moe_top1 hf_llama2_7b_moe_top2 hf_llama2_7b_hydralora hf_llama2_7b_moslora hf_llama2_7b_smear hf_llama2_7b_smora --datasets mmlu_ppl_ac766d

python run.py --models hf_llama2_7b_moe_soft hf_llama2_7b_moe_top1 hf_llama2_7b_moe_top2 hf_llama2_7b_hydralora hf_llama2_7b_moslora hf_llama2_7b_smear --datasets mmlu_ppl_ac766d


