from typing import Literal as L
from typing import reveal_type

class Suc[T]: ...

def suc[T](pred: T) -> Suc[T]:
    ...

class A:
    def __init__(self): ...

a: A = A()

a_s = a
n = 2 + 2
for i in range(n):
    a_s = suc(a_s)

reveal_type(a_s)