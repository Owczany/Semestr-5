import random
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    pipeline,
)

MODEL_NAME = "flax-community/papuGaPT2"

ANSWER_FILENAME     = 'task4_answers.txt'
QUESTION_FILENAME   = 'task4_questions.txt'

def read_files():
    with open(QUESTION_FILENAME, 'r', encoding='utf-8') as question_file:
        questions = question_file.readlines()
        questions = [q.strip() for q in questions]  

    with open(ANSWER_FILENAME, 'r', encoding='utf-8') as answer_file:
        answers = answer_file.readlines()
        answers = [a.replace('\t', ' ').strip() for a in answers]  

    return questions, answers

questions, answers = read_files()

dictionary = dict(zip(questions, answers))

q = []
a = []
for _ in range(20):
    r = random.randint(0, len(questions))
    q.append(questions[r])
    a.append(answers[r])

for i in range(len(q)):
    print(f'{i + 1}: {q[i]}')

for i in range(len(a)):
    print(f'{i + 1}: {a[i]}')
    

@dataclass
class MaskedLMEngine:
    name: str = "sdadas/polish-roberta-base-v2"
    topk: int = 10
    device: str = "auto"  # auto|cpu|cuda|mps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument()
    parser.add_argument("--in", dest="inp", default=None, help="plik z pytaniami (po 1 na linię)")
    parser.add_argument("--out", dest="out", default=None, help="plik wynikowy TSV (pytanie\todp.)")


if __name__ == "__main__":
    main()