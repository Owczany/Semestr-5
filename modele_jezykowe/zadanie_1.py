# CZAT BOT JEZA z PSEM

from collections import deque
from typing import List, Tuple

from transformers import pipeline, set_seed


MODEL_NAME = "flax-community/papuGaPT2"


generator = pipeline('text-generation', model='flax-community/papuGaPT2', device=0)

print ('Model loaded')
last_prompt = 'Dzisiaj na obiad zjemy kartofelki z'

while True:
    prompt = input().strip()
    if not prompt:
        prompt = last_prompt
    g = generator(prompt, 
       pad_token_id=generator.tokenizer.eos_token_id)[0]['generated_text']
    
    print (g)
    print (50 * '=')
    print ()
    last_prompt = prompt
    
    
    
    