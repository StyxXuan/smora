from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-2-7b-hf-chat',
        path='/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouyixiao-240108120127/work/acl-2025-moe-lora/model/Llama-2-7b-hf/',
        peft_path="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouyixiao-240108120127/work/acl-2025-moe-lora/LLaMA-Factory/results/llama2-7b-srmole-64-8-code/",
        tokenizer_path="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouyixiao-240108120127/work/acl-2025-moe-lora/LLaMA-Factory/results/llama2-7b-srmole-64-8-code/",
        max_out_len=2048,
        batch_size=16,
        torch_dtype ="torch.bfloat16",
        run_cfg=dict(num_gpus=4),
    )
]
