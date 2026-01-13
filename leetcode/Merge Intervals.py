from typing import List

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        n = len(intervals)
        sorted_intervals = sorted(intervals, key=lambda x: (x[0], x[1]))

        compressed_intervales = []

        s, e = sorted_intervals[0]
        for i in range(1, n):
            si, ei = sorted_intervals[i]

            if e < si:
                compressed_intervales.append([s, e])
                s, e = si, ei
                continue

            e = max(e, ei)

        compressed_intervales.append([s, e])

        return compressed_intervales
