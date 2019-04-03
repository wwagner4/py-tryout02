import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.close('all')


def t1():
    s = pd.Series([1, 3, 5, np.nan, 6, 8])
    a = np.array(range(1, 6))
    print("a:{0}".format(a))
    print("s:\n{0}".format(s))

def t2():
    y = np.random.randn(10)
    print(y)
    x = pd.date_range('1/1/2000', periods=10)
    print(x)
    s1 = pd.Series(y, x)
    s1.cumsum()
    s1.plot()


#t1()
#print("------------------------------------------------------------")
t2()
print("------------------------------------------------------------")
