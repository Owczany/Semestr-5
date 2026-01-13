from typing import List

# Solution with division simple
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix_prod = [1] * n
        suffix_prod = [1] * n

        prefix_prod[0] = nums[0]
        suffix_prod[-1] = nums[-1]

        for i in range(1, n):
            prefix_prod[i] = prefix_prod[i-1] * nums[i] 

        for i in range(n-2, -1, -1):
            suffix_prod[i] = suffix_prod[i+1] * nums[i]

        print(prefix_prod)
        print(suffix_prod)

        products_without_self = [suffix_prod[1]]

        for i in range(1, n-1):
            products_without_self.append(prefix_prod[i-1] * suffix_prod[i+1])
        products_without_self.append(prefix_prod[-2])

        return products_without_self

        

# Follow up without division
# class Solution:
#     def productExceptSelf(self, nums: List[int]) -> List[int]:
        