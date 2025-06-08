Title: sbintuitions/sarashina2-vision-8b · Hugging Face

URL Source: https://huggingface.co/sbintuitions/sarashina2-vision-8b

Markdown Content:
**Sarashina2-Vision-8B** is a Japanese Large Vision Language Model trained by [SB Intuitions](https://www.sbintuitions.co.jp/).

This model is based on [Sarashina2-7B](https://huggingface.co/sbintuitions/sarashina2-7b) and Image Encoder of [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B).

It achieved the highest level of scores in 4 benchmarks (as of 2025/03/07) compared to other Japanese VLMs.

[](https://huggingface.co/sbintuitions/sarashina2-vision-8b#how-to-use) How to use
----------------------------------------------------------------------------------

### [](https://huggingface.co/sbintuitions/sarashina2-vision-8b#1-install-dependencies) 1. Install dependencies

\`\`\`
pip install -U transformers==4.47.0 torch torchvision pillow protobuf sentencepiece accelerate
\`\`\`

### [](https://huggingface.co/sbintuitions/sarashina2-vision-8b#2-inference) 2. Inference

The following script loads the model and allows inference.

\`\`\`
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# Define model path
model_path = "sbintuitions/sarashina2-vision-8b"

# Load model and processor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

message = [{"role": "user", "content": "この写真に写っているもので、最も有名と考えられる建築物は何でどこに写っていますか？"}]
text_prompt = processor.apply_chat_template(message, add_generation_prompt=True)
"""text_prompt: <s><|prefix|><|file|><|suffix|>A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.

### Human: この写真に写っているもので、最も有名と考えられる建築物は何でどこに写っていますか？
### Assistant:"""

sample_image_url = "https://huggingface.co/sbintuitions/sarashina2-vision-8b/resolve/main/sample.jpg"
image = Image.open(requests.get(sample_image_url, stream=True).raw).convert("RGB")
inputs = processor(
    text=[text_prompt],
    images=[image],
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")
stopping_criteria = processor.get_stopping_criteria(["\n###"])

# Inference: Generation of the output
output_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    temperature=0.0,
    do_sample=False,
    stopping_criteria=stopping_criteria,
)
generated_ids = [
    output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
]
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
print(output_text[0])
"""この写真に写っているもので、最も有名と考えられる建築物は東京タワーです。東京タワーは、東京のランドマークであり、この写真では、高層ビル群の向こう側に写っています。"""
\`\`\`

### [](https://huggingface.co/sbintuitions/sarashina2-vision-8b#example) Example

![Image 1](https://huggingface.co/sbintuitions/sarashina2-vision-8b/resolve/main/sample.jpg)

| Prompt | Output |
| --- | --- |
| この写真に写っているもので、最も有名と考えられる建築物は何でどこに写っていますか？ | この写真に写っているもので、最も有名と考えられる建築物は東京タワーです。東京タワーは、東京のランドマークであり、この写真では、高層ビル群の向こう側に写っています。 |
| 真ん中に映っている赤と白の物は何ですか？ | 真ん中に映っている赤と白のものはクレーンです。 |

[](https://huggingface.co/sbintuitions/sarashina2-vision-8b#training) Training
------------------------------------------------------------------------------

**Sarashina2-Vision** is created through the following three-stage learning process:

1.   We tune the parameters in the projector by caption datasets.
2.   We tune the parameters in the Vision Encoder and projector by caption datasets.
3.   We tune the parameters in the projector and LLM by Visual Instruction datasets.

[](https://huggingface.co/sbintuitions/sarashina2-vision-8b#evaluation-results) Evaluation Results
--------------------------------------------------------------------------------------------------

| Model | Model Size | JMMMU*1 | Heron-Bench*2 | JDocQA |
| --- | --- | --- | --- | --- |
| [heron-chat-git-ja-stablelm-base-7b-v1](https://huggingface.co/turing-motors/heron-chat-git-ja-stablelm-base-7b-v1) | 7B | 0.294 | 0.461 | 0.069 |
| [llava-calm2-siglip](https://huggingface.co/cyberagent/llava-calm2-siglip) | 7B | 0.07 | 0.521 | 0.084 |
| [Llama-3-EvoVLM-JP-v2](https://huggingface.co/SakanaAI/Llama-3-EvoVLM-JP-v2) | 8B | 0.389 | 0.509 | 0.103 |
| [Asagi-14B](https://huggingface.co/MIL-UT/Asagi-14B) | 14B | 0.302 | 0.433 | 0.06 |
| [llm-jp-3-vila-14b](https://huggingface.co/llm-jp/llm-jp-3-vila-14b) | 14B | 0.23 | **0.665** | 0.176 |
| [EZO-InternVL2-26B](https://huggingface.co/AXCXEPT/EZO-InternVL2-26B) | 26B | 0.389 | 0.609 | 0.196 |
| [Sarashina2-Vision-8B](https://huggingface.co/sbintuitions/sarashina2-vision-8b) | 8B | 0.393 | 0.648 | 0.229 |
| [Sarashina2-Vision-14B](https://huggingface.co/sbintuitions/sarashina2-vision-14b) | 14B | **0.433** | 0.644 | **0.245** |

1.   Evaluated only single image samples (1,286 samples). If answer extraction failed, we treated it as incorrect (score 0) instead of making a random choice to eliminate stochasticity.
2.   GPT-4o (gpt-4o-2024-08-06) was used for LLM-as-a-Judge.

[](https://huggingface.co/sbintuitions/sarashina2-vision-8b#ethical-considerations-and-limitations) Ethical Considerations and Limitations
------------------------------------------------------------------------------------------------------------------------------------------

Sarashina2-Vision might generate some meaningless sequences, some inaccurate instances or biased/objectionable outputs. Before using Sarashina2-Vision, we would like developers to tune models based on human preferences and safety considerations.

[](https://huggingface.co/sbintuitions/sarashina2-vision-8b#license) LICENSE
----------------------------------------------------------------------------

[MIT License](https://huggingface.co/sbintuitions/sarashina2-vision-8b/blob/main/LICENSE)
