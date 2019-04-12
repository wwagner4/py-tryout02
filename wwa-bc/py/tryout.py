from collections import namedtuple
H = namedtuple("H", "a b c")
x = H(1, 2, True)

print(x.a)
print(x.b)
print(x.c)

