from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Step 1: Load pre-trained model
base_model = AutoModelForCausalLM.from_pretrained("/dev/shm/merge/checkpoint-500" , device_map="auto").to(
    dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("/dev/shm/merge/checkpoint-500")
print(base_model.dtype)
# Step 2: Load the LoRA model
lora_model = PeftModel.from_pretrained(base_model, "/dev/shm/checkpoint/checkpoint-400")
print(lora_model.dtype)
# Step 3: Merge LoRA into the base model
merge_model = lora_model.merge_and_unload()
print(merge_model.dtype)
# Step 4: Save the merged model
merge_model.save_pretrained("/dev/shm/teu/checkpoint-400")
tokenizer.save_pretrained("/dev/shm/teu/checkpoint-400")

#  CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /dev/shm/merge_164/checkpoint-1700 --host 0.0.0.0 --enable-auto-tool-choice --tool-call-parser hermes --tensor-parallel-size 4 --port 43560 --max-logprobs 20

# ssh -R 0.0.0.0:43560:0.0.0.0:43560 root@38.102.125.144
# ek:Y5sfm$9Zc
