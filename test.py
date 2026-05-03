import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM
import quasar 
import json


# ---- 1. Load model ----
model_name = "/root/train/model"
model = AutoModelForCausalLM.from_pretrained(model_name , 
        torch_dtype="bfloat16",
        attn_implementation="eager",
        trust_remote_code=True,
         device_map="auto",
        use_safetensors=True,)

from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=128,
    lora_alpha=1280,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    init_lora_weights="gaussian",
    use_rslora=True
)
model = get_peft_model(model, lora_config)

# model.eval()
print("eval")
class JsonlDataset(Dataset):
    def __init__(self, path):
        self.samples = []
        with open(path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        print(idx)
        return torch.tensor(self.samples[idx]["input_ids"], dtype=torch.long)

from torch.utils.data import DataLoader

def collate_fn(batch):
    return torch.nn.utils.rnn.pad_sequence(
        batch,
        batch_first=True,
        padding_value=0
    )

dataset = JsonlDataset("/root/train/dataset/teutonic_train.jsonl")
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
print("datset")

# ---- 4. Run model ----
with torch.no_grad():
    for batch in loader:
        device = next(model.parameters()).device  # safe even with device_map
        batch = batch.to(device)
        print(batch.shape)

        outputs = model(input_ids=batch)
        logits = outputs.logits
        print(torch.sum(logits))
        print("Logits shape:", logits.shape)
        break

