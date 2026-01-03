from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3-8b-instruct-hf',
        path='/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouyixiao-240108120127/work/acl-2025-moe-lora/model/Llama-3-8b-Instruct/',
        max_out_len=2048,
        batch_size=64,
        num_workers=36,
        run_cfg=dict(num_gpus=4),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    )
]
