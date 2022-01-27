import bisect

a = [(1, 10000), (2, 1000000000), (3, 10000000), (5, 999999)]
print(a)
bisect.insort(a, (4, 6), key=lambda a: a[0])
print(a)