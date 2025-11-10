import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
from collections import deque
from typing import List, Tuple

print('Ładuje rozmowę z chatem ...')

MODEL_NAME = "flax-community/papuGaPT2"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 68
MAX_TURNS_IN_PROMPT = 14
NUM_CANDIDATES = 10              
MAX_NEW_TOKENS = 50              
TEMPERATURE = 0.8
TOP_P = 0.85

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

SYSTEM_PREAMBLE = (
    "Poniżej znajduje się scenka dialogowa w stylu scenariusza filmowego.\n"
    "Rola BOT (LISTONOSZ): rozmowny, zwięzły towarzysz rozmowy. Odpowiada krótko (1–3 zdania),\n"
    "logicznie i naturalnie, trochę marzycielski, ale nie wychodzi poza temat użytkownika.\n"
    "Zawsze pisze po polsku. Formatuj linie jako 'USER:' i 'BOT:'.\n"
)

EXAMPLE_FEW_SHOT = (
    "USER: Hej! Kim jesteś?\n"
    "BOT: Jestem listonoszem, który jest w Tobie zakochany.\n"
    "USER: Opowiedz co Cię do mnie sprowadza.\n"
    "BOT: Jestem listonoszem, więc to naturalne, że przychodzę w sprawie listów oraz miłości.\n"
)

def build_prompt(history: deque) -> str:
    lines = [SYSTEM_PREAMBLE, EXAMPLE_FEW_SHOT]
    for who, text in list(history)[-MAX_TURNS_IN_PROMPT:]:
        prefix = "USER" if who == "user" else "BOT"
        text = (text or "").replace("\r", " ").strip()
        lines.append(f"{prefix}: {text}")
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
    # print(candidates)
    scored = [(score_candidate(user_utterance, c), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], scored

def main():
    set_seed(SEED)
    generator = pipeline("text-generation", model=MODEL_NAME, device=-1)
    history = deque()
    print('Aby zakończyć napisz: Nara')
    while True:
        try:
            user = input("(Ty) ŻONA: ").strip()
            # zakończenie rozmowy
            if (user.lower() == 'nara'):
                print("(BOT) Listonosz: Naraaa!")
                break
        except KeyboardInterrupt:
            print("\n(BOT) Listonosz: Tak bez pożegnania!?")
            break

        if not user:
            print("Żona uciekła z miejsca rozmowy.")
            break

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
            return_full_text=False,
        )

        raw_texts = [o["generated_text"] for o in outs]  
        trimmed = [first_sentence(t) for t in raw_texts]
        best, _ = choose_best(user, trimmed)

        # Fallback, gdyby wszystko było puste
        if not best:
            best = "Jasne, w czym mogę pomóc?"

        print(f"(BOT) Listonosz: {best}")
        history.append(("bot", best))

if __name__ == "__main__":
    main()

'''
(Ty) ŻONA: Cześć listonoszu, męża nie ma w domu. Może wejdziesz na chwilę?
(BOT) Listonosz: Załatwiaj rzeczy w domu, a jeśli chcesz, możesz się mną zająć.
(Ty) ŻONA: Uuu podoba mi się to słyszy ta kocica. Chętnie się tobą zajmę!
(BOT) Listonosz: Ta kocia koteczka jest urocza.
(Ty) ŻONA: Babcia lubi takie teksty!
(BOT) Listonosz: Babcia jest bardzo ładna, ale nie mam do niej serca.
(Ty) ŻONA: To znaczy, że mnie nie kochasz?                                
(BOT) Listonosz: To dobrze, że się nie rozstajesz.
(Ty) ŻONA: Chodzi ci o mojego męża? To o niego jesteś zazdronsy?
(BOT) Listonosz: Bardzo lubię mojego męża.
(Ty) ŻONA: Ty masz męża? Jesteś gejem?
(BOT) Listonosz: Jak chcesz, to możesz mi pomóc.
(Ty) ŻONA: Z czym ci pomóc listonoszku?
(BOT) Listonosz: Z tym, że ty kochasz mnie, a ja ciebie.
(Ty) ŻONA: Kocham cię nad życie. Chcę, żebyś teraz Ty mnie tak bardzo kochał!
(BOT) Listonosz: A teraz, dlaczego ty kochasz mnie?
(Ty) ŻONA: Bo sprawiasz, że czuję się potrzebna!
(BOT) Listonosz: Kocham cię, bo sprawiasz, że czuję się potrzebna!
(Ty) ŻONA: Teraz jesteś kobietą? Myślałem, że w spodniach masz przyrodzenie.
(BOT) Listonosz: Kocham cię, bo jesteś dla mnie ważna.
(Ty) ŻONA: Nara
(BOT) Listonosz: Naraaa!
'''
