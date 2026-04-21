import numpy as np
import matplotlib.pyplot as plt

hist = np.load('green2_ring_hist.npy')
hist2 = np.load('red2_ring_hist.npy')
hist3 = np.load('blue3_ring_hist.npy')
plt.plot(hist, label='Green Ring')
plt.plot(hist2, label='Red Ring')
plt.plot(hist3, label='Blue Ring')  
plt.xlabel('Hue')
plt.ylabel('Frequency')
plt.title('Hue Histogram of Green and Red Rings')
plt.xlim(0, 180)
plt.legend()    
plt.grid()
plt.show()
