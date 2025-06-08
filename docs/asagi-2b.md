Title: MIL-UT/Asagi-2B · Hugging Face

URL Source: https://huggingface.co/MIL-UT/Asagi-2B

Markdown Content:
[](https://huggingface.co/MIL-UT/Asagi-2B#model-details) Model Details
----------------------------------------------------------------------

### [](https://huggingface.co/MIL-UT/Asagi-2B#model-description) Model Description

This repository provides Asagi-2B, a large-scale Japanese Vision & Language Model (VLM). Asagi-2B has been trained on an extensive Japanese dataset, incorporating a diverse range of data sources.

A significant portion of the training data is synthesized using models such as the Japanese large language model ([CALM3-22B-Chat](https://huggingface.co/cyberagent/calm3-22b-chat)) and the English Vision & Language Model ([Phi3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)).

Importantly, we do not use LLMs that restrict the usage of their outputs in the license terms (e.g., GPT-4) to synthesize the training data.

| Model components | Model / Architecture | Parameters |
| --- | --- | --- |
| Vision encoder | [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) | 428M |
| Projector | 2-layer MLP | 64M |
| LLM | [llm-jp-3-1.8b-instruct](https://huggingface.co/llm-jp/llm-jp-3-1.8b-instruct) | 1.8B |

[](https://huggingface.co/MIL-UT/Asagi-2B#usage) Usage
------------------------------------------------------

### [](https://huggingface.co/MIL-UT/Asagi-2B#requirements) Requirements

\`\`\`
transformers==4.45.1
accelerate==0.34.2
torch==2.4.0
torchvision==0.19.0
\`\`\`

### [](https://huggingface.co/MIL-UT/Asagi-2B#how-to-use) How to use

\`\`\`
import requests
import torch
import transformers
from PIL import Image
from transformers import AutoModel, AutoProcessor, GenerationConfig

transformers.set_seed(42)
model_path = "MIL-UT/Asagi-2B"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModel.from_pretrained(
    model_path, trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

generation_config = GenerationConfig(
    do_sample=True,
    num_beams=5,
    max_new_tokens=256,
    temperature=0.7,
    repetition_penalty=1.5
)

prompt = ("以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n"
            "### 指示:\n<image>\nこの画像を見て、次の質問に詳細かつ具体的に答えてください。この写真はどこで撮影されたものか教えてください。また、画像の内容についても詳しく説明してください。\n\n### 応答:\n")

# sample image
sample_image_url = "https://raw.githubusercontent.com/uehara-mech/uehara-mech.github.io/refs/heads/master/images/shibuya.jpg"
image = Image.open(requests.get(sample_image_url, stream=True).raw)

inputs = processor(
    text=prompt, images=image, return_tensors="pt"
)
inputs_text = processor.tokenizer(prompt, return_tensors="pt")
inputs['input_ids'] = inputs_text['input_ids']
inputs['attention_mask'] = inputs_text['attention_mask']
for k, v in inputs.items():
    if v.dtype == torch.float32:
        inputs[k] = v.to(model.dtype)
inputs = {k: inputs[k].to(model.device) for k in inputs if k != "token_type_ids"}

generate_ids = model.generate(
    **inputs,
    generation_config=generation_config
)
generated_text = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

# do not print the prompt
if "<image>" in prompt:
    prompt = prompt.replace("<image>", " ")
generated_text = generated_text.replace(prompt, "")

print(f"Generated text: {generated_text}")

# >>> Generated text:  この写真は渋谷の交差点で撮影されたものです。写真には、夜の渋谷の街並みが写っており、多くの高層ビルが立ち並んでいます。特に目を引くのは、中央に位置する大きな白いビルで、その周囲には小さなビルが立ち並んでいます。ビルの窓からは街の明かりが漏れ、街全体が活気に満ちている様子が伝わってきます。
\`\`\`

### [](https://huggingface.co/MIL-UT/Asagi-2B#example) Example

![Image 1](https://raw.githubusercontent.com/uehara-mech/uehara-mech.github.io/refs/heads/master/images/shibuya.jpg)

| Prompt | Output |
| --- | --- |
| この画像を見て、次の質問に詳細かつ具体的に答えてください。この写真はどこで撮影されたものか教えてください。また、画像の内容についても詳しく説明してください。 | この写真は渋谷の交差点で撮影されたものです。写真には、夜の渋谷の街並みが写っており、多くの高層ビルが立ち並んでいます。特に目を引くのは、中央に位置する大きな白いビルで、その周囲には小さなビルが立ち並んでいます。ビルの窓からは街の明かりが漏れ、街全体が活気に満ちている様子が伝わってきます。 |

### [](https://huggingface.co/MIL-UT/Asagi-2B#more-examples) More Examples

Click to expand![Image 2](https://raw.githubusercontent.com/uehara-mech/uehara-mech.github.io/refs/heads/master/images/bridge.jpg)

| Prompt | Output |
| --- | --- |
| この画像を見て、次の指示に詳細かつ具体的に答えてください。この写真の内容について詳しく教えてください。 | この写真は、東京のレインボーブリッジを撮影したものです。橋は白く塗装されており、水の上に架かっています。背景には高層ビルが立ち並ぶ都市景観が広がっており、空は澄み切った青色で、白い雲が浮かんでいます。 |

![Image 3](https://raw.githubusercontent.com/uehara-mech/uehara-mech.github.io/refs/heads/master/images/tower.jpg)

| Prompt | Output |
| --- | --- |
| この画像を見て、次の質問に詳細かつ具体的に答えてください。この写真について評価してください。 | この写真は非常に低い解像度と品質のため、詳細な説明が困難です。 |

![Image 4](https://raw.githubusercontent.com/uehara-mech/uehara-mech.github.io/refs/heads/master/images/shrine.jpg)

| Prompt | Output |
| --- | --- |
| この画像を見て、次の質問に詳細かつ具体的に答えてください。この写真の神社について、細かいところまで詳しく説明してください。 | この写真の神社は、日本の伝統的な神社建築様式を反映しています。白い壁と赤い屋根のコントラストが美しく、神聖な雰囲気を醸し出しています。入り口には朱塗りの鳥居が立ち、神聖な空間への入り口を象徴しています。また、神社の前には石畳の道が続いており、参拝者を神聖な場所へと導いています。 |

![Image 5](https://raw.githubusercontent.com/uehara-mech/uehara-mech.github.io/refs/heads/master/images/garden.jpg)

| Prompt | Output |
| --- | --- |
| この画像を見て、次の指示に詳細かつ具体的に答えてください。これは日本庭園の中でも、どのような形式に分類される庭園ですか？また、その理由は何ですか？ | この日本庭園は、伝統的な日本庭園の形式に基づいており、自然との調和を重視した設計が特徴です。池や岩、石灯籠などが配置され、静かで落ち着いた雰囲気を醸し出しています。このような庭園は、日本の伝統的な美意識である「わびさび」を体現しており、自然と人間の調和を大切にする精神を体現しています。 |

![Image 6](https://raw.githubusercontent.com/uehara-mech/uehara-mech.github.io/refs/heads/master/images/slope.jpg)

| Prompt | Output |
| --- | --- |
| この画像を見て、次の質問に詳細に答えてください。この画像の場所を舞台とした小説のあらすじを書いてください。 | この画像の場所を舞台にした小説は、日本の伝統的な街並みを舞台にしています。石畳の道を進むと、古い木造建築が立ち並び、歴史を感じさせる風景が広がっています。石畳の道沿いには、緑豊かな植物が生い茂り、自然との調和を大切にする日本の美意識が感じられます。この物語は、都会の喧騒を離れ、自然と一体になれるような、静かで穏やかな雰囲気を醸し出しています。 |

[](https://huggingface.co/MIL-UT/Asagi-2B#training-details) Training Details
----------------------------------------------------------------------------

### [](https://huggingface.co/MIL-UT/Asagi-2B#training-data) Training Data

| Dataset | Preprocess | Stage1 | Stage2 | Size |
| --- | --- | --- | --- | --- |
| ROIS (Ours) | Synthesized | ✓ | ✓ | 8.4M |
| Japanese image text pairs | Synthesized | ✓ | ✓ | 4.4M |
| Wikipedia | Synthesized | ✓ | ✓ | 2.5M |
| Open Images | Translated | ✓ | ✓ | 680K |
| DCI | Translated | ✓ | ✓ | 7K |
| CommonCatalog CC-BY | Translated | ✓ | ✓ | 3.5M |
| LLaVA-Pretrain-JA |  | ✓ | ✓ | 550K |
| STAIR Captions |  | ✓ | ✓ | 410K |
| Flickr-JP |  | ✓ | ✓ | 160K |
| YJ Captions |  | ✓ | ✓ | 130K |
| Japanese Pascal |  | ✓ | ✓ | 5K |
| ArtBench | Synthesized |  | ✓ | 100K |
| GQA | Translated |  | ✓ | 1.9M |
| VQA v2 | Translated |  | ✓ | 880K |
| A-OKVQA | Translated |  | ✓ | 34K |
| OK-VQA | Translated |  | ✓ | 18K |
| Japanese Visual Genome | Translated |  | ✓ | 1.6M |
| PangeaInstruct |  |  | ✓ | 93K |

Note: ROIS (Ours) is a newly collected dataset crawled from the web specifically for this project. The dataset consists of image and raw text pairs, which are used to synthesize the training data.

[](https://huggingface.co/MIL-UT/Asagi-2B#evaluation) Evaluation
----------------------------------------------------------------

We evaluated our model using Heron-Bench, JA-VLM-Bench-in-the-Wild, and JA-VG-VQA-500. We used eval-mm library for this evaluation.

Here, models with "†" are not trained with GPT-generated data. Bold numbers indicate the best performance among all models, and underlined numbers indicate the best performance among models not trained with GPT-generated data.

| Model | LM Size | Heron-Bench (LLM (%)) | JA-VLM-Bench-In-the-Wild (ROUGE-L) | JA-VLM-Bench-In-the-Wild (LLM (/5.0)) | JA-VG-VQA-500 (ROUGE-L) | JA-VG-VQA-500 (LLM (/5.0)) |
| --- | --- | --- | --- | --- | --- | --- |
| Japanese InstructBLIP Alpha† | 7B | 14.0 | 20.8 | 2.42 | - | - |
| Japanese Stable VLM† | 7B | 24.2 | 23.3 | 2.47 | - | - |
| LLaVA-CALM2-SigLIP† | 7B | 43.3 | 47.2 | 3.15 | 17.4 | 3.21 |
| Llama-3-EvoVLM-JP-v2 | 8B | 39.3 | 41.4 | 2.92 | 23.5 | 2.96 |
| VILA-jp | 13B | **57.2** | **52.3** | **3.69** | 16.2 | 3.62 |
| Asagi-2B† | 1.8B | 44.7 | 48.8 | 3.26 | 53.7 | 3.69 |
| Asagi-4B† | 3.7B | 49.3 | 49.6 | 3.38 | 55.6 | 3.78 |
| Asagi-8B† | 7.2B | 54.7 | 49.4 | 3.45 | 56.43 | **3.84** |
| Asagi-14B† | 13B | 55.8 | 50.8 | 3.44 | **56.8** | **3.84** |
| GPT-4o | - | 87.6 | 37.6 | 3.85 | 12.1 | 3.58 |

[](https://huggingface.co/MIL-UT/Asagi-2B#risks-and-limitations) Risks and Limitations
--------------------------------------------------------------------------------------

The models released here are in the early stages of our research and development and have not been tuned to ensure outputs align with human intent and safety considerations.

[](https://huggingface.co/MIL-UT/Asagi-2B#model-card-authors) Model Card Authors
--------------------------------------------------------------------------------

Kohei Uehara

