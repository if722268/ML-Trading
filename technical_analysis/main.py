from itertools import combinations

a = [1,2,3,4,5,6]

for j in range(len(a)):
    print([i for i in combinations(a, j)])
