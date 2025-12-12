import scipy.io.wavfile as wav
# from scipy.io import wavfile
import scipy.signal as signal
import scipy.fft as fft
import numpy as np

# (a) Read Input Audio
F, X = wav.read('earhart.wav')

# (b) Normalize to [-1.0, 1.0] Float Range
amp16 = np.iinfo(np.int16).max
X     = X/amp16

# (c) Optional: Compute PSD using FFT
Xspec = fft.fft(X)
Xpsd  = np.abs(Xspec)**2 /len(Xspec)

# (d) Designa a LPF (Butterworth Type)
cutoff= 1000
b, a  = signal.butter(6, cutoff / (0.5 * F), btype='low')

# (5) Apply 0-Phase Filtering
filtered = signal.filtfilt(b, a, X)

# (6) Write Filtered Output (convert back to int16)
wav.write('out.wav', F, np.int16(filtered * amp16))
