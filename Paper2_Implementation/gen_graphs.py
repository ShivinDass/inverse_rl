import matplotlib.pyplot as plt
import numpy as np

err = np.load('errorsgw.npy',allow_pickle=True)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.plot(err)
plt.show()