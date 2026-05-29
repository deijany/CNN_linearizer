# import matplotlib
# matplotlib.use('TkAgg')  # 'Qt5Agg' or 'TkAgg', 'GTK3Agg', etc.
from myclasses.signal_processing import SNDR
import matplotlib.pyplot as plt

# Enable LaTeX rendering only when a working LaTeX installation exists.
# Falls back silently so the code runs unchanged on Colab (which has LaTeX)
# and on local machines that may not have it.
def _latex_available() -> bool:
    import shutil
    return shutil.which("latex") is not None

plt.rcParams['text.usetex'] = _latex_available()
import pickle
from scipy import signal
import numpy as np
import inspect

blue = '\033[0;34m'
green = '\033[0;32m'
cyan = '\033[0;36m'
clear = '\033[0m'

from matplotlib.ticker import MultipleLocator
# -------
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



def storate_dictionary(base_filename, dictionary):
    """
    Save dictionary base_name to a pickle file.
    Parameters:
        base_filename (str): The base path and name of the files before extension.
        dictionary (dict): The dictionary to be pickled and saved.
    Returns:
        None
    """
    pickle_filename = base_filename + ".pickle"
    # Save to pickle file
    with open(pickle_filename, "wb") as f:
        pickle.dump(dictionary, f)


def load_dictionary(pickle_filename):
    """
    Load a dictionary from a pickle file.
    Parameters:
        pickle_filename (str): The path and name of the pickle file to be loaded.
    Returns:
        dict: The dictionary loaded from the pickle file.
    """
    with open(pickle_filename, "rb") as f:
        return pickle.load(f)
      
def compute_statistics(X, V, X_hat, Q1_bits=32, Q2_bits=32, Q3_bits=32, print_on_screen=False, init_index=0, last_index=None):
    rm_DC = False
    X_quant = SNDR.quantize(X, Q1_bits)
    V = SNDR.quantize(V, Q2_bits)
    X_hat = SNDR.quantize(X_hat, Q3_bits)

    SNDR_array_X = SNDR(X[:, max(0, init_index):last_index], X_quant[:, max(0, init_index):last_index], remove_DC=rm_DC, grouping_method=None).SNDR_result
    SNDR_array_V = SNDR(X[:, max(0, init_index):last_index], V[:, max(0, init_index):last_index], remove_DC=rm_DC, grouping_method=None).SNDR_result
    SNDR_array_X_hat = SNDR(X[:, max(0, init_index):last_index], X_hat[:, max(0, init_index):last_index], remove_DC=rm_DC, grouping_method=None).SNDR_result

    SFDR_array_X = SNDR(None, X[:, max(0, init_index):last_index], remove_DC=rm_DC,
                           grouping_method=None).SFDR_result
    SFDR_array_V = SNDR(None, V[:, max(0, init_index):last_index], remove_DC=rm_DC,
                           grouping_method=None).SFDR_result
    SFDR_array_X_hat = SNDR(None, X_hat[:, max(0, init_index):last_index], remove_DC=rm_DC,
                           grouping_method=None).SFDR_result

    # print('\n')
    # print('-' * 25 + str(mode) + '-' * 25)
    if print_on_screen:
        print('X:', '          (', len(X), 'signals) '
                                           'SNDR:', 'min:', np.round(np.min(SNDR_array_X), 2), ' mean:', np.round(np.mean(SNDR_array_X), 2), ' max:', np.round(np.max(SNDR_array_X), 2))
        print('V:', '          (', len(V), 'signals) '
                                           'SNDR:', 'min:', np.round(np.min(SNDR_array_V), 2), ' mean:', np.round(np.mean(SNDR_array_V), 2), ' max:', np.round(np.max(SNDR_array_V), 2))
        print('X_hat:', '   (', len(X_hat), 'signals) '
                                           'SNDR:', 'min:', np.round(np.min(SNDR_array_X_hat), 2), ' mean:', np.round(np.mean(SNDR_array_X_hat), 2), 'max:', np.round(np.max(SNDR_array_X_hat), 2))
    return SNDR_array_X, SNDR_array_V, SNDR_array_X_hat



class SpectrumAnalyzer:
    def __init__(self):
        pass

    @staticmethod
    def get_window(window_type='Blackmanharris', length=100):
        """
        Get a window function of a specified type and length.

        Parameters:
        - window_type: str, the type of window to generate. Options are 'Blackmanharris', 'Hamming', and 'Rectangular'.
        - length: int, the length of the window.

        Returns:
        - window: numpy array, the window function.
        """
        if window_type == 'Blackmanharris':
            window = signal.windows.blackmanharris(length, sym=True)
        elif window_type == 'Hamming':
            window = signal.windows.hamming(length, sym=True)
        elif window_type == 'Rectangular':
            window = np.ones(length)
        else:
            # If an incorrect window is specified, use Blackmanharris and raise a warning
            window = signal.windows.blackmanharris(length, sym=True)
            print('Warning: Incorrect window type introduced. Continuing with Blackmanharris window.')
        return window

    @staticmethod
    def plot_frequency_domain(signal_matrix, title, window_type, save_path, x_rfft_max=None, save_fig=False):
        '''
        :param XM: Matrix with signals of size (n_signal, n_samples)
        :param title:
        :param window_type: Blackmanharris or Rectangular
        :param save_path: directory to save the plots
        :param x_rfft_max: For relative normalization
        :param save_fig:
        :return:
        '''
        fs = 1
        XM = np.copy(signal_matrix)

        # Set the figure width to 3.5 inches and height to maintain a 4:3 aspect ratio
        fig_width = 3.3  # inches
        fig_height = fig_width #* (3.0 / 4.0)  # 4:3 aspect ratio

        fig, axs = plt.subplots(nrows=len(XM), ncols=1, figsize=(fig_width, fig_height))
        for ii, X in enumerate(XM):
            axs[ii].set_ylabel(r'Amplitude spectrum (dB)', fontsize=8)
            axs[ii].set_xlabel(r'Normalized frequency', fontsize=8)
            if plt.rcParams['text.usetex']:
                axs[ii].set_title(r'\textbf{{{}}}'.format(title[ii]), fontsize=8)
            else:
                axs[ii].set_title(title[ii], fontsize=8, fontweight='bold')
            N = 16
            SNDR_lim = 6.02 * N + 1.76
            axs[ii].set_ylim((-SNDR_lim - 15, 0.5))
            axs[ii].margins(x=0)
            plt.subplots_adjust(hspace=0.63)
            for k in range(0, len(X), 1):
                font_dict = {'fontsize': 12, 'fontweight': 'bold', 'color': 'black'}
                # axs[ii].set_title(title[ii], fontdict=font_dict)
                if X.ndim == 1:
                    X = np.expand_dims(X, axis=0)

                sequence = X[k]  # Assuming you want to process each element of X
                window = SpectrumAnalyzer.get_window(window_type, len(sequence))
                vector_value = sequence * window
                rfft = np.abs(np.fft.rfft(vector_value))
                rfft_max = max(rfft)
                if x_rfft_max is None:
                    x_rfft_max = rfft_max
                p = 20 * np.log10(rfft / rfft_max)
                f = np.linspace(0, fs / 2, len(p))
                # Frequency Domain
                color = '#1f77b4'
                color = '#0000FF'
                # axs[ii].set_title(title[ii], fontdict=font_dict)
                axs[ii].plot(f / max(f), p, color=color, linewidth=0.35)
                axs[ii].set_ylim([-100, 0])
                # axs[ii].set_yticks([-80, "-", -60, "-", -40, "-", -20, "-", 0])
                axs[ii].yaxis.set_minor_locator(MultipleLocator(10))
                axs[ii].grid(True, which='Both', axis='y' , linestyle='--', linewidth=0.5)
        plt.show()



        if save_fig:
            disk_path = str(save_path)
            print('Saving figure to:', disk_path)
            plt.savefig(disk_path + '.pdf', bbox_inches='tight', pad_inches=0.05, transparent=True)
        return
        # plt.show()
