import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
POS_LORA_PATH = "analysis/model/pos_lora"
NEG_LORA_PATH = "analysis/model/neg_lora"
BETA = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512

def load_lora_model(base_name, lora_path):
    tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        load_in_4bit=True
    )
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    return tokenizer, model

def compute_log_probs(tokenizer, model, text):
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).input_ids.to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    return input_ids[:, 1:], token_log_probs

def load_lines(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

# Load models
tokenizer_p, model_p = load_lora_model(MODEL_NAME, POS_LORA_PATH)
tokenizer_n, model_n = load_lora_model(MODEL_NAME, NEG_LORA_PATH)

negative_lines = load_lines("negative_trajectory.txt")
results = []

for neg_text in tqdm(negative_lines, desc="Scoring negative responses"):
    try:
        ids_n, logpn = compute_log_probs(tokenizer_n, model_n, neg_text)
        _, logpp = compute_log_probs(tokenizer_p, model_p, neg_text)

        ids = ids_n[0].tolist()
        logpn = logpn[0].tolist()
        logpp = logpp[0].tolist()

        contrastive_scores = []
        for i in range(len(ids)):
            s_t = (1 + BETA) * logpp[i] - BETA * logpn[i]
            token = tokenizer_n.decode([ids[i]])
            contrastive_scores.append((token, s_t))

        results.append(contrastive_scores)
    except Exception as e:
        print(f"Error scoring: {e}")

# Save scores
with open("contrastive_scores.txt", "w") as f:
    for i, sample in enumerate(results):
        f.write(f"--- Example {i+1} ---\n")
        for token, score in sample:
            f.write(f"{token}\t{score:.4f}\n")
        f.write("\n")
