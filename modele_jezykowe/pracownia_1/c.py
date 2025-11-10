#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Program: qa_pl_simple.py
Opis: Odpowiada na proste pytania faktograficzne po polsku.
Wykorzystane techniki:
- Zero-shot/one-shot/few-shots na modelu generatywnym (GPT-2 po polsku: `sdadas/polish-gpt2-small`, z fallbackami),
- Heurystyka dla pytań arytmetycznych,
- Scoring w trybie prawdopodobieństw (PLL albo fill-mask softmax) dla wybranej grupy – stolice.

Uwaga: to szkic badawczy – nie ma zewnętrznej bazy wiedzy ani retrievalu.

Wymagania:
    pip install transformers torch regex sentencepiece

Sposób użycia:
    python qa_pl_simple.py --in pytania.txt --out odpowiedzi.txt
    # lub interaktywnie
    python qa_pl_simple.py
"""
from __future__ import annotations
import argparse
import math
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForMaskedLM,
    pipeline,
)

# --------------- Utils ---------------

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def sort_questions(questions: List[str]) -> List[str]:
    return sorted(q.strip() for q in questions if q.strip())


# --------------- Heurystyka: Arytmetyka ---------------
ARITH_RE = re.compile(r"^(ile to|policz|oblicz|jaki jest wynik)[:\s]*", re.I)
SAFE_EXPR_RE = re.compile(r"^[0-9\s+\-*/().,%]+$")


def try_arithmetic(q: str) -> Optional[str]:
    # Przykłady: "Ile to 2+2?", "Policz 15*(3+1)", "Ile to 50% z 200?"
    s = q.lower().strip()
    s = re.sub(r"[?]$", "", s)
    if not ARITH_RE.match(s):
        return None
    s = ARITH_RE.sub("", s)
    s = s.replace(",", ".")
    # Obsługa procentów: "a% z b" -> (a/100)*b
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*z\s*(\d+(?:\.\d+)?)", s)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        val = (a / 100.0) * b
        # jeśli dalej coś zostało, spróbuj podstawić wynik i policzyć resztę
        left = s[:m.start()] + str(val) + s[m.end():]
        s = left
    expr = re.sub(r"\s+", "", s)
    if not SAFE_EXPR_RE.match(expr):
        return None
    try:
        # Bezpieczna mini-eval: tylko operatory arytmetyczne
        val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
    except Exception:
        return None
    # Uładnij wynik
    if isinstance(val, float):
        # Usuń szum pływający, zaokrąglij rozsądnie
        if abs(val - round(val)) < 1e-9:
            return str(int(round(val)))
        return ("%.10g" % val).replace(".", ",")
    return str(val)


# --------------- Grupa: Stolice (fill-mask + scoring) ---------------
CAPITAL_PATTERNS = [
    re.compile(r"^jaka jest stolica\s+(.+?)\?$", re.I),
    re.compile(r"^stolica\s+(.+?)\?$", re.I),
    re.compile(r"^stolicą\s+(.+?)\s+jest\s*\??$", re.I),
]

@dataclass
class MaskedLMEngine:
    name: str = "sdadas/polish-roberta-base-v2"  # model PL RoBERTa
    topk: int = 10

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.name)
        self.fill = pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer, top_k=self.topk)
        # RoBERTa używa <mask> tokena
        self.mask = self.tokenizer.mask_token

    def ask_capital(self, country_raw: str) -> Optional[str]:
        country = normalize_whitespace(country_raw)
        # Znormalizuj nazwy typu "Polski" -> "Polski"; dodaj wielką literę
        if country and country[0].islower():
            country = country[0].upper() + country[1:]
        # Budujemy zdanie klauzowe z maską
        template = f"Stolicą {country} jest {self.mask}."
        try:
            preds = self.fill(template)
        except Exception:
            return None
        # Preferuj wyrazy zaczynające się wielką literą i bez spacji
        candidates = [p for p in preds if p.get("token_str", "").strip()]
        candidates.sort(key=lambda p: (-(p["score"]),
                                       0 if p["token_str"].strip()[:1].istitle() else 1,
                                       len(p["token_str"].strip())))
        if not candidates:
            return None
        answer = candidates[0]["token_str"].strip()
        # Oczyść z BPE artefaktów, jeśli wystąpią
        answer = answer.replace("Ġ", "").replace("▁", "")
        return answer


def try_capital(q: str, mlm: MaskedLMEngine) -> Optional[str]:
    qs = normalize_whitespace(q)
    for pat in CAPITAL_PATTERNS:
        m = pat.match(qs)
        if m:
            country = m.group(1)
            # Usuń przyimki: "włoch" vs "Włoch" etc.
            country = re.sub(r"^(państwa|kraju|krajem|w|we|z|ze)\s+", "", country, flags=re.I)
            # Usuń końcowe kropki itp.
            country = country.strip(" .")
            ans = mlm.ask_capital(country)
            return ans
    return None


# --------------- Scoring PLL (opcjonalny) ---------------
@dataclass
class PLLEncoder:
    name: str = "sdadas/polish-roberta-base-v2"

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.name)
        self.model.eval()

    @torch.no_grad()
    def pll(self, text: str) -> float:
        # Pseudo Log-Likelihood: maskujemy każde słowo i sumujemy log-proby
        tok = self.tokenizer(text, return_tensors="pt")
        input_ids = tok["input_ids"][0]
        attn = tok["attention_mask"][0]
        log_prob_sum = 0.0
        count = 0
        for i in range(1, len(input_ids)-1):  # pomijamy BOS/EOS jeśli występują
            if attn[i].item() == 0:
                continue
            masked = input_ids.clone()
            masked[i] = self.tokenizer.mask_token_id
            outputs = self.model(input_ids=masked.unsqueeze(0))
            logits = outputs.logits[0, i]
            log_probs = logits.log_softmax(dim=-1)
            log_prob_sum += log_probs[input_ids[i]].item()
            count += 1
        return log_prob_sum / max(count, 1)


# --------------- Zero/One/Few-shot: generacja ---------------
FEW_SHOT_EXAMPLES = [
    ("Ile dni ma tydzień?", "7"),
    ("Jakie jest największe morze śródlądowe na świecie?", "Morze Kaspijskie"),
    ("Jak nazywa się stolica Hiszpanii?", "Madryt"),
]

@dataclass
class GPT2Gen:
    # Lista działających identyfikatorów modeli (najpierw lekkie)
    candidates: Tuple[str, ...] = (
        "sdadas/polish-gpt2-small",
        "sdadas/polish-gpt2-medium",
        "sdadas/polish-gpt2-large",
        "sdadas/polish-gpt2-xl",
        "flax-community/papuGaPT2",
        # awaryjnie angielski GPT-2, jeśli PL niedostępne
        "openai-community/gpt2",
    )
    max_new_tokens: int = 16
    chosen_name: Optional[str] = None

    def __post_init__(self):
        last_err = None
        for name in self.candidates:
            try:
                tok = AutoTokenizer.from_pretrained(name)
                mdl = AutoModelForCausalLM.from_pretrained(name)
                # sukces
                self.tokenizer = tok
                self.model = mdl
                self.chosen_name = name
                break
            except Exception as e:
                last_err = e
                continue
        if not hasattr(self, "model"):
            raise RuntimeError(
                f"Nie udało się załadować żadnego modelu GPT-2 z listy {self.candidates}. Ostatni błąd: {last_err}"
            )

    def build_prompt(self, question: str, k: int = 2) -> str:
        k = min(k, len(FEW_SHOT_EXAMPLES))
        shots = "".join([f"Pytanie: {q} Odpowiedź: {a}" for q, a in FEW_SHOT_EXAMPLES[:k]])
        return f"{shots} Pytanie: {question} Odpowiedź: "

    @torch.no_grad()
    def generate(self, question: str, k_shots: int = 2) -> str:
        prompt = self.build_prompt(question, k_shots)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        # Ustal sensowny EOS - preferuj tokenizer.eos_token_id, a jeśli brak, użyj kodu dla 

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is None:
            newline_ids = self.tokenizer.encode("")
            eos_id = newline_ids[0] if newline_ids else None
        out = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=eos_id,
        )
        gen = self.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        gen = gen.strip()
        # Utnij na kropce/znaku zapytania/nowej linii, żeby nie halucynować dalej
        gen = re.split(r"[]|[.?!]", gen)[0].strip()
        return gen


# --------------- Router ---------------
@dataclass
class Router:
    mlm: Optional[MaskedLMEngine] = None
    pll_enc: Optional[PLLEncoder] = None
    gen: Optional[GPT2Gen] = None

    def ensure(self):
        if self.mlm is None:
            self.mlm = MaskedLMEngine()
        if self.pll_enc is None:
            self.pll_enc = PLLEncoder()
        if self.gen is None:
            self.gen = GPT2Gen()

    def answer_one(self, q: str) -> str:
        q_clean = normalize_whitespace(q)
        # 1) Heurystyka: arytmetyka
        a = try_arithmetic(q_clean)
        if a is not None:
            return a
        # 2) Stolice – fill-mask + scoring
        self.ensure()
        a = try_capital(q_clean, self.mlm)
        if a is not None:
            return a
        # 3) Yes/No scoring (opcjonalnie): wybór między "Tak." i "Nie." używając PLL
        if q_clean.lower().startswith("czy "):
            hyp_yes = q_clean + " Tak."
            hyp_no = q_clean + " Nie."
            s_yes = self.pll_enc.pll(hyp_yes)
            s_no = self.pll_enc.pll(hyp_no)
            return "Tak" if s_yes > s_no else "Nie"
        # 4) Few-shot na GPT-2 po polsku (fallback)
        return self.gen.generate(q_clean, k_shots=2) or "nie wiem"

    def answer_many(self, questions: List[str]) -> List[Tuple[str, str]]:
        questions = sort_questions(questions)
        out = []
        for q in questions:
            try:
                ans = self.answer_one(q)
            except Exception as e:
                ans = f"[błąd: {e}]"
            out.append((q, ans))
        return out


# --------------- I/O ---------------

def read_questions(path: Optional[str]) -> List[str]:
    if path is None:
        print("Wpisuj pytania (Ctrl+D aby zakończyć):", file=sys.stderr)
        data = sys.stdin.read()
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
    # Każda linia = jedno pytanie
    qs = [line.strip() for line in data.splitlines() if line.strip()]
    return qs


def write_answers(path: Optional[str], pairs: List[Tuple[str, str]]):
    lines = [f"{q}\t{a}" for q, a in pairs]
    out = "\n".join(lines)
    if path is None:
        print(out)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default=None, help="plik z pytaniami (po 1 na linię)")
    parser.add_argument("--out", dest="out", default=None, help="plik wynikowy TSV (pytanie\todp.)")
    args = parser.parse_args()

    qs = read_questions(args.inp)
    router = Router()
    pairs = router.answer_many(qs)
    write_answers(args.out, pairs)


if __name__ == "__main__":
    main()
