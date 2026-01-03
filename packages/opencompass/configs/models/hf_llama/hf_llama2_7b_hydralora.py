from opencompass.models import HuggingFacewithChatTemplate, HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='llama-2-7b-hf-hydralora-mixed-data',
        path='/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouyixiao-240108120127/work/acl-2025-moe-lora/model/Llama-2-7b-hf',
        peft_path="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouyixiao-240108120127/work/acl-2025-moe-lora/LLaMA-Factory/results/exp_res_mutidomain/llama2-7b-hydralora-mix-data",
        tokenizer_path="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouyixiao-240108120127/work/acl-2025-moe-lora/LLaMA-Factory/results/exp_res_mutidomain/llama2-7b-hydralora-mix-data",
        # generation_kwargs=dict(
        #     # num_return_sequences=10,
        #     do_sample=True,
        #     top_p=0.95,
        #     temperature=0.8,
        # ),
        max_out_len=1024,
        batch_size=8,
        # num_worker=16,
        run_cfg=dict(num_gpus=4),
    )
]
