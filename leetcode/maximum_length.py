from typing import List

def maximumLength(nums: List[int]) -> int:
    n = len(nums)
    a, b = 0, 0

    # Zamieniamy na 0 lub 1
    for i in range(n):
        nums[i] %= 2

    # Zliczanie parzysych
    seen = [0, 0]
    for i in range(n):
        seen[nums[i]] += 1

        if seen[nums[i]] > 1:
            a += 1
            seen = [0, 0]
    
    seen = nums[0]
    for i in range(1, n):
        
        if nums[i] != seen:
            seen = nums[i]
            b += 1

    return max(a, b) + 1
        

        

print(maximumLength([1, 2, 3]))