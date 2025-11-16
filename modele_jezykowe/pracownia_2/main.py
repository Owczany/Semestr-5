import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import math

MODEL_NAME = "flax-community/papuGaPT2"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# Lista prefiksów (wczytywana ze skosa)
PREFIXES = [
    "Prawdziwy piekarz przyprawia pieczywo pieprzem",
    "Mały marynarz maluje mapy morskie",
    "Dzielny drwal dźwiga duże drewno"
]

# Parametry generacji
TOP_K = 40
TOP_P = 0.9
NUM_SAMPLES = 5
MAX_NEW_TOKENS = 40


def modify_logits_for_letter_constraint(logits, tokenizer, letter):
    """Obniża logity tokenów, które zaczynają nowe słowo inną literą."""
    forbidden_penalty = -20.0

    for tok_id in range(len(logits)):
        tok_str = tokenizer.decode([tok_id])

        # token zaczyna nowe słowo (zaczyna się od spacji + litera)
        if tok_str.startswith(" "):
            stripped = tok_str.strip()
            if len(stripped) > 0 and stripped[0].lower() != letter.lower():
                logits[tok_id] += forbidden_penalty

    return logits


def sample_sequence(prefix, letter_constraint):
    input_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    generated = input_ids
    logprobs = []

    for _ in range(MAX_NEW_TOKENS):
        outputs = model(generated)
        logits = outputs.logits[:, -1, :]
        logits = logits[0].clone()

        # modyfikacja rozkładu
        logits = modify_logits_for_letter_constraint(logits, tokenizer, letter_constraint)

        # top-k
        top_k = TOP_K
        values, indices = torch.topk(logits, top_k)
        logits_filtered = torch.full_like(logits, -float("inf"))
        logits_filtered[indices] = logits[indices]

        # top-p
        sorted_logits, sorted_indices = torch.sort(logits_filtered, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        cutoff = (cumulative > TOP_P).nonzero()
        if len(cutoff) > 0:
            first_index = cutoff[0]
            sorted_logits[first_index + 1:] = -float("inf")

        # sampling
        final_logits = torch.zeros_like(logits_filtered)
        final_logits[sorted_indices] = sorted_logits

        probabilities = torch.softmax(final_logits, dim=-1)
        token_id = torch.multinomial(probabilities, 1)

        logprob = torch.log(probabilities[token_id])
        logprobs.append(logprob.item())

        generated = torch.cat((generated, token_id.unsqueeze(0)), dim=1)

        # zakończ, jeśli mamy kropkę lub inny kończący znak
        decoded = tokenizer.decode(token_id)
        if decoded in [".", "!", "?"]:
            break

    text = tokenizer.decode(generated[0])
    return text, logprobs


def score_generation(text, logprobs, letter):
    avg_logprob = sum(logprobs) / len(logprobs)

    # kara za spadki prawdopodobieństwa
    drop_penalty = 0
    for i in range(1, len(logprobs)):
        if logprobs[i] < logprobs[i - 1] - 3:
            drop_penalty += 2

    # kara za powtórzenia
    words = text.split()
    repeated_penalty = len(words) - len(set(words))

    # kara za słowa niezgodne z literą
    letter_penalty = sum(1 for w in words if w[0].lower() != letter.lower())

    score = avg_logprob - 0.5 * drop_penalty - repeated_penalty - 2 * letter_penalty
    return score


def generate_best_sentence():
    prefix = random.choice(PREFIXES)
    first_letter = prefix.split()[0][0].lower()

    candidates = []
    for _ in range(NUM_SAMPLES):
        text, logprobs = sample_sequence(prefix, first_letter)
        s = score_generation(text, logprobs, first_letter)
        candidates.append((s, text))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# Uruchom
print(generate_best_sentence())
