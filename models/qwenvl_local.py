import torch
from pathlib import Path
CURRENT_DIR = Path(__file__).parent
WORK_DIR = Path(__file__).parent.parent
import sys
sys.path.append(str(WORK_DIR))

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
# from models_utils.qwen_vl_utils import process_vision_info
from qwen_vl_utils import process_vision_info

def get_msg(image_path, prompt):
    return [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file:///{image_path}",
                },
                {"type": "text", "text": prompt},
            ],
        }]
    
class HuggingfaceQwenVL:
    def __init__(self, model_name = "Qwen/Qwen2-VL-2B-Instruct"):
        self.model, self.processor = self.get_huggingface_qwenvl(model_name)

    def get_huggingface_qwenvl(self, model_name):
        # default: Load the model on the available device(s)
        # model = Qwen2VLForConditionalGeneration.from_pretrained(
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        # default processer
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor
    
    def __call__(self, image_path, prompt, **kwargs):
        messages = get_msg(image_path, prompt)
        
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text

if __name__ == '__main__':
    # 如果需要量化非量化模型, 参考 https://huggingface.co/docs/transformers/main/tasks/image_text_to_text#fit-models-in-smaller-hardware
    LocalModel = "Qwen/Qwen2-VL-2B-Instruct" 
    # Qwen/Qwen2.5-VL-3B-Instruct  # Qwen/Qwen2.5-VL-72B-Instruct
    # Qwen/Qwen2-VL-2B-Instruct # Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4
    from prompts.nurvid_prompts import NurVidPromptGenerator
    model = HuggingfaceQwenVL(model_name=LocalModel)
    prompt_generator = NurVidPromptGenerator(annotation_dir="data/nurvid/annotations")
    prompt = prompt_generator(round_id=1)
    print(model(image_path=WORK_DIR / "data/nurvid/ajDGW_bOm5Y/frames/frame0001.jpg", prompt=prompt))