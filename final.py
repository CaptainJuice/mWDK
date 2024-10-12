import numpy as np


def f(nums1,nums2):
    mystack = []
    res = [-1] * len(nums2)
    for i in range(len(nums2)):
        while mystack and nums2[i] > nums2[mystack[-1]]:
            p = mystack.pop()
            res[nums2[p] - 1] = nums2[i]
        mystack.append(i)
    return [res[i - 1] for i in nums1]

"""
[[0 0 0 1 1 1 0 1 0 0]
 [1 1 0 0 0 1 0 1 1 1]
 [0 0 0 1 1 1 0 1 0 0]
 [0 1 1 0 0 0 1 0 1 0]
 [0 1 1 1 1 1 0 0 1 0]
 [0 0 1 0 1 1 1 1 0 1]
 [0 1 1 0 0 0 1 1 1 1]
 [0 0 1 0 0 1 0 1 0 1]
 [1 0 1 0 1 1 0 0 0 0]
 [0 0 0 0 1 1 0 0 0 1]]
"""

nums1 = [3,1,5,7,9,2,6]
nums2 =[1,2,3,5,6,7,9,11]
print(f(nums1,nums2))