import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import time

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1.1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1.1")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

prompt = """The image depicts an emergency scene on a road where a bicycle accident has occurred. Here is a detailed description:
A person is lying on the ground, seemingly injured. They are wearing a helmet and a dark-colored outfit. Next to the person, there is a fallen bicycle"""
description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that has no background noise."

s = time.time()

input_ids = description_tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()

print(time.time() - s)

sf.write("sample.wav", audio_arr, model.config.sampling_rate)
