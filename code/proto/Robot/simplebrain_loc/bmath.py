from random import randint, random, shuffle
from numpy.random import normal, uniform
from numpy import absolute

def Vcl(f1, v1, f2, v2):
    return [f1 * v1[i] + f2 * v2[i] for i in range(min(len(v1), len(v2)))]

def Vdiff(v1, v2):
    return [v1[i] - v2[i] for i in range(min(len(v1), len(v2)))]

def Vadd(v1, v2):
    return [v1[i] + v2[i] for i in range(min(len(v1), len(v2)))]

def Vmul(v, f):
    return [value * f for value in v]

def VxV(v1, v2):
    return [v1[i] * v2[i] for i in range(min(len(v1), len(v2)))]

def proba(percentage):
    return randint(1, 1000000) <= 1000000 * percentage/100

def rndChoose(L):
    return L[rndInt(0, len(L) - 1)]

def rndInt(a, b) -> int :
    return int(a + round(random() * (b - a)))

def ShuffledOf(L):
    L_shuffled = L[:]
    shuffle(L_shuffled)
    return L_shuffled

def s(x): return 1 if x > 0 else (-1 if x < 0 else 0)

def normalise(v):
    return Vmul(v, inv(norm(v)))

def inv(x): return 1/x if x != 0 else 0

def norm(x):
    return sum([v**2 for v in x])**0.5

def distance(x1, x2):
    return norm(Vdiff(x2, x1))