from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import time
import torchvision

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)


processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", use_fast = True,
)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "VizWiz_train_00014902.jpg",
#             },
#             {"type": "text", 
#             "text": """Answer the question in full details and if you can't answer the question then recommend 
#                         the user on the steps required to take the picture better such that you can answer the question.
#                         # QUESTION
#                         I'm trying to determine what this jar is. It looks like pasta sauce, but I can't tell. I got a mystery bag of groceries in the grocery delivery yesterday and I'm trying to figure out what these things are.
#                         """},
#         ],
#     }
# ]

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "directional/images/VizWiz_val_00000984.jpg",
            },
            {"type": "text", 
            "text": """A blind user took the following photo and asked this question:
                       “We need to know what is on the screen and which item is for driver signing.”
                       Based on the image and the question, should they move the camera 'left, right, up, down, or none of the above' to better capture the information needed to answer it? Respond with only one direction.
                        """},
        ],
    }
]

s = time.time()
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
print(output_text[0])
print(time.time() - s)
