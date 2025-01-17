import matplotlib.pyplot as plt
from training_loop import losses
plt.figure()
plt.plot(losses)
plt.xlabel("Steps")
plt.ylabel("Avg Loss")
plt.show()