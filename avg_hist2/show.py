import numpy as np
import matplotlib.pyplot as plt

hist = np.load('green_average.npy')
hist2 = np.load('red_average.npy')
hist3 = np.load('blue_average.npy')
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
