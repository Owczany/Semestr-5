from collections import deque
# from types import List

def jump(nums) -> int:
    n = len(nums)
    visited = [0 for _ in range(n)]
    dp = [ 0 for _ in range(n)]

    q = deque()
    q.append((0, 0))
    visited[0] = 1

    while q:
        x, min_step = q.popleft()
        max_steps = nums[x]

        for i in range(1, max_steps + 1):
            if x + i >= n:
                break
            if visited[x + i]:
                continue

            dp[x + i] = min_step + 1

            visited[x + i] = 1
            q.append((x + i, min_step + 1))
    
    return dp[-1]


print(jump([2,3,0,1,4]))