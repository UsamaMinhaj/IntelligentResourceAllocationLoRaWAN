# from .loratools import PRR_calculator
import loratools
import numpy as np
import channel
#import node
from numpy.random import Generator, PCG64, SeedSequence
import math
sg = SeedSequence(1234)
x = np.array([-17.2, -14.2, -11.2, -8.2, -5.2, -2.2])

PRR = loratools.PRR_calculator(x, 7, 5, 125, 40)
print(np.round(PRR, 3))

g_d1 = 2.7e-14
rg = Generator(PCG64(12345))
rg = [Generator(PCG64(s)) for s in sg.spawn(10)]

mu = 1
Omega = 1
size = 1
for x in range(10):
    h1 = math.sqrt(rg[x].gamma(mu, Omega / mu, size))
    output_signal = h1 * g_d1
    print(f"{np.round(h1, 3)} + {output_signal}")
output_signal = h1 * g_d1

print(output_signal)

#y = channel.nakagami(g_d1, 5, 1, 1, node.myNode.rg)
#print(y)
