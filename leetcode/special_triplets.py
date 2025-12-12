from collections import Counter
from typing import List

def specialTriplets(nums: List[int]) -> int:
    n = len(nums)
    MOD = 10**9 + 7
    rc = Counter(nums[2:])  # Right Counter
    lc = Counter(nums[:1])  # Left Counter

    res = 0

    # O(n)
    for i in range(1, n-1):
        target = nums[i] << 1
        res = (res + (lc[target] * rc[target])) % MOD
        lc[nums[i]] += 1
        rc[nums[i+1]] -= 1
    
    return res
            
# Sprawdzaczka
test_cases = [[6, 3, 6], [1, 2, 3], [3, 2, 1]]
for test_case in test_cases:
    nums = test_case
    print(specialTriplets(nums))