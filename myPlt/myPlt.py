import matplotlib.pyplot as plt
import numpy as np

x = np.arange(150) * 0.073434
y = np.sin(x)

print(x)
print(y)

plt.plot(x, y)
plt.show()

