from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import torchvision, os, json


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", use_fast = True,
)

# Path to the JSON file
json_file_path = "directional/Directional_Guidance_annotation_v2%28recommended%29.json"  # Change to your actual file path

# Load JSON data
with open(json_file_path, 'r') as f:
    data = json.load(f)

output_dict = dict()

for d in tqdm(data):

    question = d['Question']
    image = d['Image']
    direction = d['Directional_guidance']

    try:

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": os.path.join("directional/images/", image)
                    },
                    {"type": "text", 
                    "text": f"""Given the image and question you MUST tell user to MOVE the camera in one of the following direction given in #DIRECTION CHOICES 
                                so that the question can be answered.
                                # QUESTION
                                {question}
                                # DIRECTION CHOICES
                                ['leave it unchanged', 'left', 'right', 'up', down']
                            """},
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
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output = output_text[0].lower().strip()

        temp = {
            "question": question,
            "answer": output,
            "direction": direction,
        }
        output_dict[image.split(".")[0]] = temp
    
    except:
        print(f"Error in image {image}")
        temp = {
            "question": question,
            "answer": "Error",
            "direction": direction,
        }
        output_dict[image.split(".")[0]] = temp
    
    print(output_dict)

    break