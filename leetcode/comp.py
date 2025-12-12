from collections import Counter
from typing import List
from math import factorial

def countPermutations( complexity: List[int]) -> int:
        n = len(complexity)
        root = complexity[0]
        
        for i in range(1, n):
              if complexity[i] <= root:
                    return 0


        return factorial(n-1)