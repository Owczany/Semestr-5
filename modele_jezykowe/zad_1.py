
import re
import math
from collections import deque
from typing import List, Tuple

from transformers import pipeline, set_seed

# =============== KONFIG ===============
MODEL_NAME = "flax-community/papuGaPT2"
SEED = 33
MAX_TURNS_IN_PROMPT = 6          # ile ostatnich wypowiedzi (USER/BOT) trzymamy w promptcie
NUM_CANDIDATES = 6               # ile odpowiedzi generujemy naraz
MAX_NEW_TOKENS = 100             # górny limit generacji (potem i tak przycinamy)
TEMPERATURE = 0.5
TOP_P = 0.92
# ======================================

# --------- PROMPT BAZOWY (rola + styl) ---------
SYSTEM_PREAMBLE = (
    "Poniżej znajduje się scenka dialogowa w stylu scenariusza filmowego z bajki 'Kot w Butach'\n"
    "Rola BOT: Jest śmiercią, która walczy z Rola USER: który jest kotem w butach\n",
    # "Rola BOT: rozmowny, zwięzły towarzysz rozmowy. Odpowiada krótko (1-2 zdania),\n"
    "logicznie i naturalnie, bez rozwlekłości. Nie wychodzi poza temat użytkownika.\n"
    "Zawsze pisze po polsku. Formatuj linie jako 'USER:' i 'BOT:'.\n"
)

EXAMPLE_FEW_SHOT = (
    "USER: Uwazaj z kim walczysz przeklęty\n",
    "BOT: Siemka jestem twoim koszmarem\n",
    "USER: Hej! Kim jesteś?\n"
    "BOT: Jestem lekkim czatbotem do pogaduch - mogę odpowiadać zwięźle i rzeczowo.\n"
    "USER: Opowiedz suchara.\n"
    "BOT: Informatyk nie płacze - ma tylko problem z wilgotnością w oczach.\n"
)

def build_prompt(history: deque) -> str:
    lines = [SYSTEM_PREAMBLE, EXAMPLE_FEW_SHOT]
    for who, text in list(history)[-MAX_TURNS_IN_PROMPT:]:
        prefix = "USER" if who == "user" else "BOT"
        lines.append(f"{prefix}: {text.strip()}")
    lines.append("BOT:")
    return "\n".join(lines)

def first_sentence(text: str) -> str:
    text = text.split("\nUSER:")[0].strip()
    if "BOT:" in text:
        text = text.split("BOT:", 1)[1].strip()
    import re as _re
    m = _re.search(r"([^.?!]{2,}[.?!])", text)
    out = m.group(1).strip() if m else text.strip()
    return out[:200].strip()

def repetition_ratio(text: str) -> float:
    import re as _re
    tokens = [t for t in _re.findall(r"\w+", text.lower()) if t]
    if not tokens:
        return 1.0
    unique = len(set(tokens))
    return unique / len(tokens)

def jaccard_sim(a: str, b: str) -> float:
    import re as _re
    A = set(_re.findall(r"\w+", a.lower()))
    B = set(_re.findall(r"\w+", b.lower()))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def score_candidate(user_utterance: str, candidate: str) -> float:
    import re as _re
    txt = candidate.strip()
    length = len(txt)
    len_score = -0.002 * max(0, length - 120) + (0.2 if 25 <= length <= 160 else 0.0)
    rep = repetition_ratio(txt)
    rep_score = 0.6 * rep
    end_score = 0.2 if _re.search(r"[.?!]$", txt) else 0.0
    jac = jaccard_sim(user_utterance, txt)
    topic_score = 0.6 * jac
    role_penalty = -0.5 if "USER:" in txt or "BOT:" in txt[5:] else 0.0
    return len_score + rep_score + end_score + topic_score + role_penalty

def choose_best(user_utterance: str, candidates: List[str]) -> Tuple[str, List[Tuple[float, str]]]:
    scored = [(score_candidate(user_utterance, c), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], scored

def main():
    set_seed(SEED)
    generator = pipeline("text-generation", model=MODEL_NAME, device=-1)
    print("Model loaded.")
    print("Napisz coś. Pusta linia powtórzy ostatnie pytanie. Ctrl+C aby wyjść.")
    history = deque()
    last_user = "Cześć, o czym pogadamy?"
    while True:
        try:
            user = input("> ").strip()
        except KeyboardInterrupt:
            print("\nDo zobaczenia!")
            break
        if not user:
            user = last_user
        last_user = user
        history.append(("user", user))

        prompt = build_prompt(history)
        outs = generator(
            prompt,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_return_sequences=NUM_CANDIDATES,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=generator.tokenizer.eos_token_id,
        )

        raw_texts = [o["generated_text"] for o in outs]
        trimmed = [first_sentence(t) for t in raw_texts]
        best, scored = choose_best(user, trimmed)

        print(best)
        print("=" * 60)
        history.append(("bot", best))

if __name__ == "__main__":
    main()
