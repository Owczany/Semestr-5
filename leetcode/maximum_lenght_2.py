 

def maximumLength(nums: List[int], k: int) -> int:
        n = len(nums)
        res = 0

        for i in range(n):
            nums[i] %= k
        
        c = Counter(nums)

        s = c.most_common(1)[0][0]
        print(c[s])

maximumLength([1,2,3,4,5,6, 4], 4)
        