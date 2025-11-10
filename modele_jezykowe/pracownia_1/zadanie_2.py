import itertools
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import functional as F

MODEL_NAME = "flax-community/papuGaPT2"
DEVICE = 'cpu'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

def log_probs_from_logits(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logp_label
    
def sentence_prob(sentence_txt: str) -> float:
    input_ids = tokenizer(sentence_txt, return_tensors='pt')['input_ids'].to(DEVICE)
    with torch.no_grad():
        output = model(input_ids=input_ids)
        log_probs = log_probs_from_logits(output.logits[:, :-1, :], input_ids[:, 1:])
        seq_log_probs = torch.sum(log_probs)
    return seq_log_probs.cpu().numpy()  

def generate_permutations(words):
    results = []
    for perm in itertools.permutations(words):
        sent = " ".join(perm)
        sent = sent[0].upper() + sent[1:] + "."
        results.append(sent)
    return results

def create_words(sentence: str) -> list:
    return sentence.lower().replace(".", "").replace(",", "").split()

sentences = [
    'Babuleńka miała dwa rogate koziołki.',
    'Wiewiórki w parku zaczepiają przechodniów.',
    'Wejdę kupić jedno piwo czeskie',
    'Dzieci wesoło wybiegły ze szkoły',
    'Lata lecą a bilety jakoś nie są ju ulgowe.'
]

for s in sentences:
    words = create_words(s)
    sentences = generate_permutations(words)

    scored = [(s, sentence_prob(s)) for s in sentences]
    scored.sort(key=lambda x: x[1], reverse=True)

    for s, score in scored[:5]:
        print(f"{s}  [score={score:.3f}]")

    for s, score in scored[-5:]:
        print(f"{s}  [score={score:.3f}]")
