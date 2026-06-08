from transformers import AutoModelForCausalLM
import torch

model_id = "/workspace/text_to_image/tencent_HunyuanImage_3/pretrained_model"
assert torch.cuda.is_available(), "CUDA broken"

kwargs = dict(
    attn_implementation="sdpa",   # keep stable attention
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,   # DO NOT use "auto"
    device_map="auto",

    # 🚀 THIS IS THE KEY CHANGE
    moe_impl="flashinfer",
)

model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

model.load_tokenizer(model_id)

prompt = "A brown and white dog is running on the grass"

with torch.inference_mode():
    image = model.generate_image(
        prompt=prompt,
        stream=False,   # ⚠️ flashinfer is more stable without streaming
    )

image.save("/workspace/text_to_image/tencent_HunyuanImage_3/output_image/image.png")
print("Done")