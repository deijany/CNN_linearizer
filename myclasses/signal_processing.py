import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import firwin, freqz, lfilter
import matplotlib.pyplot as plt


import inspect

blue = '\033[0;34m'
green = '\033[0;32m'
cyan = '\033[0;36m'
clear = '\033[0m'
def who_is_calling():
    current_frame = inspect.currentframe()
    # The function's caller
    parent_frame = current_frame.f_back
    parent_info = inspect.getframeinfo(parent_frame)

    # function container
    caller_frame = parent_frame.f_back
    info = inspect.getframeinfo(caller_frame)

    # Handle cases where the function is called from the top-level, function or class
    if caller_frame.f_code.co_name == "<module>":
        script_name = caller_frame.f_code.co_filename
        print(f"{cyan}{parent_info.function}{clear} is called at the {cyan}top-level{clear} of the script {script_name}")
    elif 'self' in caller_frame.f_locals:
        class_name = caller_frame.f_locals['self'].__class__.__name__
        print(f"{cyan}{parent_info.function}{clear} is called by the method {green}{info.function}{clear} of class {blue}{class_name}{clear}")
    else:
        print(f"{cyan}{parent_info.function}{clear} is called by the function {green}{info.function}{clear}")
class FIRFilter:
    """
    FIRBandpass: Finite Impulse Response (FIR) Filter

    This class represents a finite impulse response bandpass filter with specified parameters.
    It includes methods to obtain the impulse response, filter signals, and plot the frequency
    and impulse responses.

    Parameters:
    - order (int): Order of the filter.
    - fs (float): Sampling frequency.
    - wc1 (float): Lower cutoff frequency.
    - wc2 (float): Upper cutoff frequency.
    - width (float): Width of the pass-band.

    Methods:
    - get_impulse_response(): Get the impulse response of the filter.
    - filter_signal(signals, impulse_response): Filter signals using the provided impulse response. It admits (M,M), (M,V) & (V;V)
    - plot_frequency_response(): Plot the frequency response of the filter.
    - plot_impulse_response(): Plot the impulse response of the filter.
    """

    def __init__(self, order, fs, wc, width, pass_zero):
        self.order = order
        self.fs = fs
        self.wc = np.array(wc)
        self.width = width
        self.pass_zero = pass_zero

    def get_impulse_response(self):
        return firwin(self.order + 1, self.wc, self.width, pass_zero=self.pass_zero, fs=self.fs)


    @staticmethod
    def filter_signal(signals, impulse_response, axis = 0):
        # Check if signals is a 2D array
        if axis == 1:
            signals = signals.T

        if signals.ndim == 2:
            if impulse_response.ndim == 2:
                # Convolve each signal with each impulse response
                result = np.array(
                    [np.convolve(signal, impulse, mode='same') for signal, impulse in zip(signals, impulse_response)])
            else:
                # Convolve each signal with the single impulse response
                result = np.array([np.convolve(signal, impulse_response, mode='same') for signal in signals])
        else:
            # Convolve the single signal with the impulse response
            result = np.convolve(signals, impulse_response, mode='same')
        return result if axis == 0 else result.T

    def plot_frequency_response(self, G):
        h = self.get_impulse_response()
        w, H = freqz(h, [1], fs=self.fs, worN=2000)
        plt.figure()
        plt.clf()
        plt.plot(w / np.max(w), 20 * np.log10(abs(H) / np.max(abs(H), axis=0)))
        plt.title("Frequency Response")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.show()

    def plot_impulse_response(self, G):
        h = self.get_impulse_response()
        impulse = np.zeros((self.order + 1))
        impulse[0] = 1  # Create an impulse signal
        response = lfilter(h, [1], impulse)
        plt.figure()
        plt.stem(response, basefmt='k-')  #older python versions  use_line_collection=True
        plt.title("Impulse Response")
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True)
    plt.show()

class Interpolator:
    """
    A class for performing signal interpolation and downsampling operations.
    This class provides methods for upsampling and downsampling 1D signals.

    Attributes: None
    """
    def __init__(self):
        pass

    @staticmethod
    def upsample_signal(X, upsample_factor, kind='linear', axis=0):
        """
        Upsample a 1D signal or multiple 1D signals by a specified factor using linear interpolation or zero-padding.

        Parameters:
        - X: numpy array, the original signal(s) as a 1D or 2D array (MxN, where M is the number of signals and N is the length of each signal).
        - upsample_factor: int, the factor by which to upsample the signal(s).
        - kind: str, interpolation method ('linear' or 'zeros').

        Returns:
        - upsampled_signal(s): numpy array, the upsampled signal(s) as a 2D array.
        """
        if upsample_factor == 1:
            return X

        if axis == 1:
            X = X.T

        return_one_dim = False
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
            return_one_dim = True
        M, N = X.shape
        if kind == 'linear':
            upsampled_signal = np.zeros((M, N * upsample_factor - 1))
            for ii, x in enumerate(X):
                # Create an interpolation function
                interp_func = interp1d(np.arange(len(x)), x, kind)
                # Generate new indices for the upsampled signal
                new_indices = np.arange(0, len(x) - 1 + 1 / upsample_factor, 1 / upsample_factor)
                # Use the interpolation function to upsample the signal
                upsampled_signal[ii, :] = interp_func(new_indices)
        elif kind == 'zeros':
            upsampled_signal = np.zeros((M, N * upsample_factor-1))
            upsampled_signal[:, ::upsample_factor] = X
        else:
            raise ValueError("Invalid interpolation kind. Use 'linear' or 'zeros'.")

        if return_one_dim:
            result = upsampled_signal[0]
        else:
            result = upsampled_signal
        return result.T if axis == 1 else result


    @staticmethod
    def downsample_signal(X, downsample_factor, axis = 0):
        """
        Downsample a 1D signal or multiple 1D signals by a specified factor using decimation.

        Parameters:
        - X: numpy array, the original signal(s) as a 1D or 2D array (MxN, where M is the number of signals and N is the length of each signal).
        - downsample_factor: int, the factor by which to downsample the signal(s).

        Returns:
        - downsampled_signal(s): numpy array, the downsampled signal(s) as a 1D or 2D array.
        """
        if downsample_factor==1:
            return X
        if axis == 1:
            X = X.T
        return_one_dim = False
        if X.ndim == 1:
            return_one_dim = True
            X = np.expand_dims(X, axis=0)
        M, N = X.shape
        downsampled_length = N // downsample_factor
        downsampled_signal = np.zeros((M, downsampled_length+1))
        for ii, x in enumerate(X):
            # Perform downsampling by selecting every 'downsample_factor'-th sample
            downsampled_signal[ii, :] = x[::downsample_factor]

        if return_one_dim:
            result = downsampled_signal[0]
        else:
            result = downsampled_signal
        return result.T if axis == 1 else result
    @classmethod
    def example_usage(cls):
        """
        Example usage of the Interpolator class.

        This class method demonstrates how to use the Interpolator class to upsample and downsample a signal.
        """
        original_signal = np.sort(np.random.rand(1,5))
        # print('original signal:', original_signal.shape, original_signal)
        upsample_factor = 2

        upsampled_signal = cls().upsample_signal(original_signal, upsample_factor, kind='linear')
        # print('upsampled signal:', upsampled_signal.shape, upsampled_signal)

        downsample_factor = 2
        downsampled_signal = cls().downsample_signal(upsampled_signal, downsample_factor)
        # print('downsampled signal:', downsampled_signal.shape, downsampled_signal)



# class SNDR:
#     def __init__(self, X_ref, X_est, remove_DC=True, grouping_method='mean'):
#         self.remove_DC = remove_DC
#         self.X_ref = X_ref
#         self.X_est = X_est
#         self.grouping_method = grouping_method
#         if X_ref.ndim == 1:
#             X_ref = np.expand_dims(X_ref, axis=0)
#             X_est = np.expand_dims(X_est, axis=0)
#             self.result = self._simple_signal(X_ref, X_est)
#         elif X_ref.ndim == 2 and len(X_ref) == 1:
#             self.result = self._simple_signal(X_ref, X_est)
#         elif X_ref.ndim == 2 and len(X_ref) >= 1:
#             self.result = self._multiple_signal(X_ref, X_est)
#         else:
#             raise ValueError('Input dimensions not supported')
#
#     def _simple_signal(self, X_ref, X_est):
#         if self.remove_DC:
#             X_est = X_est - np.mean(X_est)
#         num = np.sum(np.abs(X_ref)**2)
#         den = np.sum(np.abs(X_est - X_ref)**2)
#         SNDR = 10 * np.log10(num / den)
#         return SNDR
#
#     def _multiple_signal(self, X_ref, X_est):
#         SNDR_est = []
#         for position in range(0, len(X_ref), 1):
#             x = np.expand_dims(X_ref[position], axis=0)
#             x_est = np.expand_dims(X_est[position], axis=0)
#             SNDR_est.append(self._simple_signal(x, x_est))
#         if self.grouping_method == 'min':
#             SNDR = np.min(SNDR_est)
#         elif self.grouping_method == 'max':
#             SNDR = np.max(SNDR_est)
#         elif self.grouping_method == 'mean':
#             SNDR = np.mean(SNDR_est)
#         else:
#             SNDR = SNDR_est
#         return SNDR
#
#     @staticmethod
#     def quantize(signals, amount_bits):
#         # Assumes a full-scale signal x in the interval [-1,1)
#         Q = 2**-(amount_bits-1)
#         signals = signals * (1-1e-12)
#         signals = signals - Q/2
#         signals = np.round(signals * 2**(amount_bits-1)) / 2**(amount_bits-1)
#         return signals + Q/2

import numpy as np

import numpy as np


class SNDR:
    def __init__(self, X_ref=None, X_est=None, remove_DC=True, grouping_method='mean'):
        self.remove_DC = remove_DC
        self.grouping_method = grouping_method

        if X_ref is not None and X_est is not None:
            self.SNDR_result = self.compute_snr(X_ref, X_est)
        else:
            self.SNDR_result = None

        if X_est is not None:
            self.SFDR_result = self.compute_sfdr(X_est)
        else:
            self.SFDR_result = None

    def compute_snr(self, X_ref, X_est):
        if X_ref.ndim == 1:
            X_ref = np.expand_dims(X_ref, axis=0)
            X_est = np.expand_dims(X_est, axis=0)
            return self._simple_signal_snr(X_ref, X_est)
        elif X_ref.ndim == 2 and len(X_ref) == 1:
            return self._simple_signal_snr(X_ref, X_est)
        elif X_ref.ndim == 2 and len(X_ref) >= 1:
            return self._multiple_signal_snr(X_ref, X_est)
        else:
            raise ValueError('Input dimensions not supported')

    def _simple_signal_snr(self, X_ref, X_est):
        if self.remove_DC:
            X_est = X_est - np.mean(X_est)
        num = np.sum(np.abs(X_ref) ** 2)
        den = np.sum(np.abs(X_est - X_ref) ** 2)
        SNDR = 10 * np.log10(num / den)
        return SNDR

    def _multiple_signal_snr(self, X_ref, X_est):
        SNDR_est = []
        for position in range(len(X_ref)):
            x = np.expand_dims(X_ref[position], axis=0)
            x_est = np.expand_dims(X_est[position], axis=0)
            SNDR_est.append(self._simple_signal_snr(x, x_est))
        return self._aggregate_results(SNDR_est)

    def compute_sfdr(self, X):
        if self.remove_DC:
            X = X - np.mean(X)
        if X.ndim == 1:
            X_est = np.expand_dims(X, axis=0)
            return self._simple_signal_sfdr(X)
        elif X.ndim == 2 and len(X) == 1:
            return self._simple_signal_sfdr(X)
        elif X.ndim == 2 and len(X) >= 1:
            return self._multiple_signal_sfdr(X)
        else:
            raise ValueError('Input dimensions not supported')

    def _simple_signal_sfdr(self, X_est, threshold_db=0, margin=300):
        """
        Compute the Spurious-Free Dynamic Range (SFDR) for a given signal.

        Parameters:
        X_est (np.ndarray): Input signal.
        threshold_db (float): Threshold in dB to ignore peaks within this range of the fundamental.
        margin (int): Number of samples to exclude around each fundamental peak.

        Returns:
        float: SFDR value in dB.
        """
        N = X_est.shape[1]
        fft_signal = np.fft.fft(X_est, axis=1)
        amplitude_spectrum = 2.0 / N * np.abs(fft_signal[:, :N // 2])

        # Convert amplitude spectrum to dB
        amplitude_spectrum_db = 20 * np.log10(amplitude_spectrum + np.finfo(float).eps)  # Add eps to avoid log(0)

        # Identify peaks within 5 dB of 0 dB as fundamentals
        is_fundamental = amplitude_spectrum_db >= -threshold_db

        # Create a mask to zero out the fundamentals and the margin around them
        mask = np.zeros_like(amplitude_spectrum_db, dtype=bool)
        for i in range(is_fundamental.shape[1]):
            if is_fundamental[0, i]:
                start = max(0, i - margin)
                end = min(is_fundamental.shape[1], i + margin + 1)
                mask[:, start:end] = True
        # print(mask)

        # Set the fundamentals and the margin around them to zero to find the largest spurious peak
        amplitude_spectrum_db[mask] = -np.inf

        # Find the largest spurious peak
        largest_spurious_amplitude_db = np.max(amplitude_spectrum_db, axis=1)

        # Compute SFDR
        sfdr = -largest_spurious_amplitude_db  # Since fundamentals are at 0 dB
        return sfdr

    # Integrate this function into the class as before

    def _multiple_signal_sfdr(self, X_est):
        SFDR_est = []
        for position in range(len(X_est)):
            x_est = np.expand_dims(X_est[position], axis=0)
            SFDR_est.append(self._simple_signal_sfdr(x_est))
        return self._aggregate_results(SFDR_est)

    def _aggregate_results(self, results):
        if self.grouping_method == 'min':
            result = np.min(results)
        elif self.grouping_method == 'max':
            result = np.max(results)
        elif self.grouping_method == 'mean':
            result = np.mean(results)
        else:
            result = results
        return result

    @staticmethod
    def quantize(signals, amount_bits):
        QUE = 2 ** -(amount_bits - 1)
        signals = signals * (1 - 1e-12)
        signals = signals - QUE / 2
        signals = np.round(signals * 2 ** (amount_bits - 1)) / 2 ** (amount_bits - 1)
        return signals + QUE / 2


class VectorNormalizer:
    @staticmethod
    def z_score(data, axis=0):
        """Apply Z-Score normalization to the data."""
        data_np = np.array(data)
        mean_val = data_np.mean(axis=axis)
        std_val = data_np.std(axis=axis)
        return (data_np - mean_val) / std_val

    @staticmethod
    def min_max_scaling(data, new_min=0, new_max=1, axis=0):
        """Apply feature scaling to the data."""
        data_np = np.array(data)
        min_val = data_np.min(axis=axis)
        max_val = data_np.max(axis=axis)
        return new_min + (data_np - min_val) * (new_max - new_min) / (max_val - min_val)

if __name__ == "__main__":
    Interpolator.example_usage()