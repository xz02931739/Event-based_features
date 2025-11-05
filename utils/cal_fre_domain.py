import numpy as np
from scipy.stats import skew, kurtosis
from scipy import signal
from matplotlib import pyplot as plt
from scipy.integrate import simps
import antropy as ant


class FrequencyDomain_tool():
    def __init__(self, x, fs, switch=False) -> None:
        self.x = x
        self.fs = fs
        self.psd_calculate(300, switch)

    def psd_calculate(self, win_len_in_seconds, switch):
        win = win_len_in_seconds * self.fs
        self.freqs, self.psd = signal.welch(self.x, self.fs, nperseg=win, noverlap=((1/4)* win))
        if switch is False:
            pass
        else:
            print (self.freqs)
            
    def band_power(self, band_range):
        # plt.semilogy(freqs, psd, color='k', lw=2)
        # plt.xlim([0, 0.05])
        # plt.ylim([0, psd.max() * 1.1])

        low, high = band_range[0], band_range[-1]
        idx_delta = np.logical_and(self.freqs >= low, self.freqs <= high)
        # plt.fill_between(self.freqs, self.psd, where=idx_delta, color='skyblue')

        freq_res = self.freqs[idx_delta]
        delta_x = freq_res[-1] - freq_res[0]
        power = simps(self.psd[idx_delta], dx=delta_x)

        peak_f = max(self.psd[idx_delta])

        return power, peak_f


    def spectralentropy(self):
        return ant.spectral_entropy(self.psd, self.fs)
    
    def statistical_moments_of_psd(self):
        first_moments = np.mean(self.psd)
        second_moments = np.var(self.psd)
        third_moments = kurtosis(self.psd)
        fourth_moments = skew(self.psd)

        return first_moments, second_moments, third_moments, fourth_moments