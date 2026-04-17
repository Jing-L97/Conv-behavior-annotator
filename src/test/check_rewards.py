import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---- CHANGE THIS ----
model_path = "/scratch2/jliu/Feedback/models/reward/cr/checkpoint-2650"

# load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()


def get_logits(sentences):
    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(-1)

    return logits.tolist()


# ---- Example usage ----
sentences = [
    "What had some girl revealed she hasn't?",  # trigger
    "no sister could carry at least ten windows.",  # NOT trigger cr
]

logits = get_logits(sentences)
logits
