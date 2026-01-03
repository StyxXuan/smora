from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama3-8b-srmole-64-8-python-code-18k',
        path='/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouyixiao-240108120127/work/acl-2025-moe-lora/model/Llama-3-8b-Instruct/',
        peft_path="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouyixiao-240108120127/work/acl-2025-moe-lora/LLaMA-Factory/results/llama3-8b-srmole-64-8-python-code-18k",
        tokenizer_path="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouyixiao-240108120127/work/acl-2025-moe-lora/LLaMA-Factory/results/llama3-8b-srmole-64-8-python-code-18k",
        generation_kwargs=dict(
            num_return_sequences=1,
            do_sample=True,
            top_p=0.95,
            temperature=0.2,
        ),
        max_out_len=2048,
        batch_size=16,
        num_workers=36,
        run_cfg=dict(num_gpus=4),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    )
]
