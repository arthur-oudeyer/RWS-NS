from math import exp, tanh

_memo_sig = {}
def sigmoid(x):
    if x not in _memo_sig:
        if x < -50: return 0.
        elif x > 50: return 1.
        else:
            try:
                _memo_sig[x] = 1/(1 + exp(-x))
            except:
                raise ValueError(f"Out of range : {x}")

    return _memo_sig[x]

_next_id = 0
def getNewId():
    global _next_id
    _next_id += 1
    return _next_id
