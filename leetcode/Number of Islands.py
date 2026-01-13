from collections import deque
from typing import List

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]:
            return 0

        m, n = len(grid), len(grid[0])
        visited = set()
        islands = 0
        dirs = [(0,1), (1,0), (-1,0), (0,-1)]

        for i in range(m):
            for j in range(n):
                if grid[i][j] != '1' or (i, j) in visited:
                    continue

                islands += 1
                visited.add((i, j))
                q = deque([(i, j)])

                while q:
                    ci, cj = q.popleft()
                    for di, dj in dirs:
                        ni, nj = ci + di, cj + dj
                        if (0 <= ni < m and 0 <= nj < n and
                            grid[ni][nj] == '1' and
                            (ni, nj) not in visited):
                            visited.add((ni, nj))     # mark on enqueue
                            q.append((ni, nj))

        return islands
