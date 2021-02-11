Learned Wavelet Filterbanks
=== 

### Overview

This project explores the possibility of a deep learning model to jointly learn:
  - a bank of **wavelets**, or set of signal "fragments", that can be linearly combined to produce arbitrary signals   
  - the **weights** for that linear combination, given a particular signal window

    
### Motivation
  
These wavelets can be used as features for downstream audio tasks, like classification.
  
The weights produced for a particular audio signal are similar to the power spectrum of a time-frequency transform, 
so they can be used similarly as high level features of the audio signal.


### Visualization

![Signal reconstructions](./media/reconstructed4.jpg)

---

An example learned filterbank:

![Spectrogram](./media/wavelet7.jpg)

![](./media/w5.png)

--- 

Plotting the weights against a FFT spectrogram. Notice how because of the non-linearity of the analysis, you can 
zoom in much further and achieve better time-frequency specificity than FFT, although 'frequency' is approximated.

Also note how sparse the 'learned wavelet transform' is compared to FFT, enabling high compression.  

![](./media/spectrogram6.jpg)

## 