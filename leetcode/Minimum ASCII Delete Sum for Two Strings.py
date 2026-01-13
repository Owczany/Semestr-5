from typing import List

class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        n, m = len(s1), len(s2)
        # To nam tworzy tablicę m x n
        lcs = [[0 for _ in range(n+1)] for _ in range(m+1)]
        
        '''
            x1 x2 x3 ... xn    s1
        y1  
        y2
        y3
        .
        .
        .
        ym

        s2
        '''

        for i in range(m):
            for j in range(n):
                if s1[j] == s2[i]:
                    lcs[i+1][j+1] = (2 * ord(s1[j])) + lcs[i][j]
                else:
                    lcs[i+1][j+1] = max(lcs[i+1][j], lcs[i][j+1])

        a = 0

        for letter in s1:
            a += ord(letter)
        for letter in s2:
            a += ord(letter)

        print(a)
        print(lcs)

        return a -  lcs[-1][-1]
    
Solution.minimumDeleteSum(None, "delete", "leet")