from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import time
from kokoro import KPipeline
from faster_whisper import WhisperModel
import soundfile as sf

# default: Load the model on the available device(s)
model_vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)

min_pixels = 256 * 28 * 28
max_pixels = 768 * 28 * 28
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels, use_fast = True,
)

pipeline_tts = KPipeline(lang_code='a')

model_stt = WhisperModel("distil-small.en")

def get_vlm_output(user_input : str = "Describe this", image_path : str = "accident.jpeg", 
                    video_path : str = "space_woaudio.mp4", image_flag : bool = True) -> str:

    default_prompt = f"""
                        Assume a bling person is infront this scene. 
                        Answer this question.
                        {user_input}
                        Then, You must give a warning if there is any important warning, potential obstruction or any danger present. 
                    """
    if image_flag:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "resized_height": 240,
                        "resized_width": 240,
                    },
                    {"type": "text", "text": default_prompt},
                ],
            }
        ]
    
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": 0.01,
                        "max_pixels": 240 * 320,
                    },
                    {"type": "text", "text": default_prompt},
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
    generated_ids = model_vlm.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    print("VLM Time ", time.time() - s)

    return output_text[0].replace("\n", "")


def get_tts_output(input: str = "Hello, I am a retard."):
    s = time.time()
    generator = pipeline_tts(
        input, voice='af_heart', # <= change voice here
        speed=1,
    )
    for i, (gs, ps, audio) in enumerate(generator):
        sf.write(f'output.wav', audio, 24000) # save each audio file
    print("TTS Time ", time.time() - s)

def get_stt_output(sound_file_path : str = "input.wav"):
    s = time.time()
    segments, info = model_stt.transcribe(sound_file_path)
    output = ""
    for segment in segments:
        output += segment.text
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    print("STT Time ", time.time() - s)
    return output


def main():
    question = input("Enter question (default: 'Describe the scene'): ") or "Describe the scene"
    audio_file_path = input("Enter audio file path : ") or None
    image_path = input("Enter image file path : ") or None
    video_path = input("Enter video file path : ") or None

    if audio_file_path is not None:
        question = get_stt_output(audio_file_path)
    
    if image_path is not None:
        vlm_out = get_vlm_output(question, image_path)
        get_tts_output(vlm_out)
    
    elif video_path is not None:
        vlm_out = get_vlm_output(question, video_path = video_path, image_flag = False)
        get_tts_output(vlm_out)

    else:
        vlm_out = get_vlm_output(question)
        get_tts_output(vlm_out)


if __name__ == "__main__":
    main()

