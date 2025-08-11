import xml.etree.ElementTree as ET
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import torchvision, os, json


# Load and parse the XML file
tree = ET.parse('archive/test.xml')  # Replace with your actual file path
root = tree.getroot()

image_tags_dict = dict()

# Loop through each image element
for image in root.findall('image'):
    image_path = image.find('imageName').text
    
    image_tags_dict[image_path] = []
    
    tagged_rectangles = image.find('taggedRectangles')
    
    if tagged_rectangles is not None:
        for rectangle in tagged_rectangles.findall('taggedRectangle'):
            tag = rectangle.find('tag').text
            image_tags_dict[image_path].append(tag)


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", use_fast = True,
)

output_dict = dict()

for image, tags in tqdm(image_tags_dict.items()):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": os.path.join("archive", image),
                },
                {"type": "text", "text": "Output all the text you see in this image. Do not skip over any text visible"},
            ],
        }
    ]


    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=64)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    output = output_text[0].replace('\n', ' ').lower().strip()
    tag = " ".join(tags).lower().strip()

    output_dict[f"{image}_tag"] = tag
    output_dict[f"{image}_Qwen/Qwen2.5-VL-7B-Instruct"] = output

with open('output_Qwen2.5-VL-7B-Instruct.json', 'w') as f:
    json.dump(output_dict, f, indent=4)


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained(
    "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit", use_fast = True,
)

output_dict = dict()

for image, tags in tqdm(image_tags_dict.items()):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": os.path.join("archive", image),
                },
                {"type": "text", "text": "Output all the text you see in this image. Do not skip over any text visible"},
            ],
        }
    ]


    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=64)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    output = output_text[0].replace('\n', ' ').lower().strip()
    tag = " ".join(tags).lower().strip()

    output_dict[f"{image}_tag"] = tag
    output_dict[f"{image}_unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit"] = output

with open('output_Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit', 'w') as f:
    json.dump(output_dict, f, indent=4)

