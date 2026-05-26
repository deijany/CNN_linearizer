### Frequency Dependent Linearizer


import numpy as np
from myclasses.signal_processing import FIRFilter, Interpolator
from myfunctions.functions_helper import who_is_calling, SpectrumAnalyzer

white = '\033[1;37m'
orange ='\033[38;5;208m'

ANSI_colors = [
    '\033[0;34m',  # blue
    '\033[0;36m',  # cyan
    '\033[0;32m',  # green
    '\033[0;35m',  # purple
    '\033[0;31m',  # red
    '\033[1;37m'   # white
]
original_colors = ANSI_colors.copy()
for _ in range(5):
    ANSI_colors.extend(original_colors)

clear='\033[0m'

import time
import random
import itertools

# import matplotlib
# matplotlib.use('TkAgg')

# import copy
def calculate_sfdr(reference, test):
    # Perform FFT on both signals
    X = np.fft.fft(reference)
    Y = np.fft.fft(test)

    # Calculate power spectral density or simply use the amplitude spectrum
    X_psd = np.abs(X)
    Y_psd = np.abs(Y)

    # Find the peak signal (fundamental frequency) - excluding DC component if present
    fundamental_idx = np.argmax(X_psd[1:]) + 1  # +1 to correct for zero-indexing and skipping DC

    # Find the maximum spurious signal in the test signal
    Y_spur = np.max(np.delete(Y_psd, fundamental_idx))  # Delete the fundamental frequency

    # Calculate SFDR
    sfdr = 20 * np.log10(X_psd[fundamental_idx] / Y_spur)
    return sfdr

class MatrixInversionLinearizer:
    """
    Matrix Inversion Linearizer for Signal Processing.

    Attributes:
        X_ref (numpy.ndarray): Reference signal(s).
        V (numpy.ndarray): Input signal(s).
        branch_number (int): Number of parallel branches in the network.
        activation_function (ActivationFunctions): Activation function (or non-linnearity) used in the network.
        number_simulations (int): Number of simulations to find the best solution (it means, number of bias generations).
        L2_constant (float): Constant used to regularize the matrix inversion.
        linearizer_order (int): Memory order of the linearizer (related to number of blocks bult-ed on the linearizer) (default order is 0).
        output_quantizer_resolution (int): Resolution for output quantization inside the branches (default is 50).
        model_name (str): Name of the model (default is 'noname').
        best_SNR (float): Best Signal-to-Noise Ratio achieved during training (initial value per default is -inf).
        reuse_bias (bool): Flag to reuse bias during training on each block (default is False).
        static_filter (FIRFilter): Static filter to apply during prediction (default is None).

    Methods:
        _bkgenerator(seed, mode):
            Generate bias vectors (bk) based on the specified mode.

        _delay_sequence(v, current_order):
            Get previous samples based on the current order.

        _make_matrix(v, BK):
            Helper function to create the matrix used for the .

        _NMSE(y_true, y_pred, mode='min'):
            Calculate the mean root square error (NMSE) between true and predicted signals.

        _wk_solver(current_BK):
            Helper function to solve for the weights (ak) in the network. It is the core of the matrix inversion based linearizer.

        train():
            Train the network by finding the best weights and biases.

        nested_loops(number_simulations, linearizer_order):
            Generate combinations of nested loops for simulations that do not re-use bias in each block.

        predictor(X_ref, V, ak, bk):
            Predict the output based on the input and weights.
    """
    first_call = True

    def __init__(self, X_ref, V, branch_number, activation_functions, number_simulations, L2_constant, linearizer_order=0, causal_system = False, output_quantizer_resolution=50, model_name='noname', best_error=np.inf, reuse_bias=False,
                 static_filter=None):
        # Initialize the object with the following attributes:
        # - X_ref: reference signal(s)
        # - V: input signal(s)
        # - branch_number: number of branches in the network
        # - activation_function: activation function used in the network
        # - number_simulations: number of simulations to run to find the best solution
        # - L2_constant: constant used to regularize the matrix inversion
        first_call = True

        self.X_ref = X_ref
        self.V = V
        self.branch_number = branch_number  # these that going to activation function
        self.activation_function = activation_functions[0]
        # self.original_activation_functions = activation_functions
        # self.activation_functions_iterator = itertools.cycle(activation_functions)
        self.number_simulations = number_simulations
        self.L2_constant = L2_constant
        self.output_quantizer_resolution = output_quantizer_resolution
        self.linearizer_order = int(linearizer_order)
        self.causal_system=causal_system
        self.model_name = model_name
        self.best_error = best_error

        self.reuse_bias = reuse_bias
        self.static_filter = static_filter  #impulse_response


    def _bkgenerator(self, seed, mode):
        # Generate bias vectors (bk) based on the specified mode
        if mode == 'random':
            np.random.seed(seed + int(time.time()))
            bk = 2 * (np.random.rand(self.branch_number) - 0.5)

        elif mode == 'linesearch':
            a = -1.0
            b = 1.0
            step = 0.25 * (b - a) / self.number_simulations  # calculate the step size
            a = np.round(a + seed * step, 6)  # Adjust a for current iteration
            b = np.round(b - seed * step, 6)  # Adjust b for current iteration
            bk = np.linspace(a, b, self.branch_number, endpoint=True)
        else:
            raise ValueError('Invalid bk generator method')
        bk = np.expand_dims(bk, 1)
        if self.activation_function == ActivationFunctions.polynomial:
            bk = abs(bk) * 0
        return bk

    @staticmethod
    def _delay_sequence(x, delay, axis=0):
        # who_is_calling()
        v = np.copy(x)
        if v.ndim==1:
            v = np.expand_dims(v, axis=0)
        if axis == 0:  # horizontal delay
            # v_slice = v[:, delay:]
            if delay >= 0:
                slice = v[:, delay:]
                last = np.zeros((v.shape[0], delay))
                v_slice = np.concatenate((slice, last), axis=1)
            else:
                slice = v[:, :delay]
                init = np.zeros((v.shape[0], abs(delay)))
                v_slice = np.concatenate((init,slice), axis=1)

        elif axis == 1: # vertical delay
            # For axis 1, shift elements down in the row (when transposed, this looks like a column shift), padding with zeros at the top
            if delay >= 0:
                v_slice = v[delay:, :]
                last = np.zeros((delay, v.shape[1]))
                v_slice = np.concatenate((v_slice, last), axis=0)
            else:
                v_slice = v[:delay, :]
                init = np.zeros((abs(delay), v.shape[1]))
                v_slice = np.concatenate((init, v_slice), axis=0)

        if x.ndim==1:
            return v_slice[0]
        else:
            return v_slice

    import numpy as np

    def _toeplitz_matrix_from_sequence(self, x, order, mode=None):
        """
        Generate a Toeplitz matrix for the sequence x, suitable for convolution with the sequence h.
        Parameters:
        - x: Input sequence as a list or 1D NumPy array.
        - h: Kernel (filter) sequence as a list or 1D NumPy array.
        Returns:
        - Toeplitz matrix as a 2D NumPy array shape:(order+1, len(x)/(len(x)+order/2)).
        """
        L = len(x[0])
        K = order + 1
        # Size of the resulting convolution
        size = L + K - 1
        # Initialize the Toeplitz matrix with zeros
        toeplitz_matrix = np.zeros((K, size))
        # Populate the Toeplitz matrix
        if order == 0:
            return x
        else:
            for col in range(size):
                for row in range(K):
                    if 0 <= col - row < L:
                        toeplitz_matrix[row, col] = x[0, col - row]
            if mode == 'same':
                return toeplitz_matrix[:, order//2:-order//2]
            else:
                return toeplitz_matrix

    def _make_matrix(self, v, BK, interpolation='False'):
    # def _make_matrix(self, v, BK, interpolation='True'):
        # Helper function to create the matrix used in the network
        vin= np.copy(v.T)
        if interpolation == 'True':
            ups_factor = 1
            # filter --------
            width = 0.01
            ups_filter = FIRFilter(order=200, fs=2, wc=[(1 - width) / ups_factor], width=width / ups_factor, pass_zero='lowpass')

            G = ups_factor
            h_ups = ups_filter.get_impulse_response()
            h_ups = G*h_ups
            # if MatrixInversionLinearizer.first_call:
            #     ups_filter.plot_frequency_response(G)
            #     ups_filter.plot_impulse_response(G)
            #     MatrixInversionLinearizer.first_call = False

            vin_ups = Interpolator().upsample_signal(vin, upsample_factor=ups_factor, kind='zeros', axis=1)
            # ------
            ur_input = ups_filter.filter_signal(vin_ups, h_ups, axis=1)
        else:
            ups_factor = 1
            ur_input = vin

        Ar_c0 = np.ones(np.shape(vin))
        ur_c1 = np.copy(vin)

        for current_order in range(0, self.linearizer_order + 1):
            #---- Causality
            if self.causal_system:
                delay = current_order   #remove for causal
            else:
                if current_order % 2 == 0:
                    delay = current_order // 2
                else:
                    delay = -(1 + current_order // 2)
            delay*=ups_factor
            #----
            ur_bypass_prev = self._delay_sequence(ur_c1, delay, axis=1)
            ur_prev = self._delay_sequence(ur_input, delay, axis=1)
            # print('ur_bypass_prev', ur_bypass_prev.shape, 'ur_prev', ur_prev.shape)
            #-----

            if BK.shape[1] == 1:
                bk = BK
            else:
                bk = np.expand_dims(BK[:, current_order], axis=1)
            total_branch = len(bk)
            if self.activation_function == ActivationFunctions.polynomial:
                for branch_iter in range(0, total_branch):
                    result = self.activation_function(ur_prev, power=branch_iter + 2, axis=1)
                    result = Interpolator().downsample_signal(result, downsample_factor=ups_factor, axis=1)
                    if branch_iter == 0:
                        Ark_ups = result  # (Lx1)
                    else:
                        Ark_ups = np.concatenate([Ark_ups, result], axis=1)  # (LxK)
            else:
                result= self.activation_function(ur_prev+bk.T) # remove
                Ark_ups = self.activation_function(ur_prev+bk.T) # 2D-array

            if current_order == 0:
                Ar = Ark_ups
                Ar_c1 = ur_bypass_prev
            else:
                Ar = np.concatenate([Ar, Ark_ups], axis=1)
                Ar_c1 = np.concatenate([Ar_c1, ur_bypass_prev], axis=1)

        Ar = np.concatenate([Ar, Ar_c1, Ar_c0], axis=1)  #, Ar_c1, vin_c0
        # Ar = ups_filter.filter_signal(Ar, h_ups, axis=1)

        yy=1
        # print('Ar_nor:',Ar[0:8,yy])
        # Ar = Interpolator().downsample_signal(Ar, downsample_factor=ups_factor, axis=1)
        # print('Ar_dow:', Ar[0:8, yy])

        NLB = total_branch+1
        Arc_numb = total_branch*(self.linearizer_order+1)
        colBranch = NLB*self.linearizer_order

        # print('Ar:', np.shape(Ar), 'Ar_c0:', np.shape(Ar_c0), 'ur_prev:', np.shape(ur_prev), 'Bches: [', NLB,',', colBranch, '] l.order:', self.linearizer_order)
        # print(f"{white}ur_prev:{ur_input[0:5,0]}{clear}")
        # current_color = -1
        # for iii in range(0,len(Ar[0,:])):
        #     if (iii) % (total_branch) == 0:  # Change the color every 3 iterations.
        #         current_color +=1
        #     if iii == len(Ar[0,:])-1:
        #         print(f"{white}Ar_c0{Ar[0:5, iii]}{clear}")
        #     elif iii < len(Ar[0,:])-1 - (self.linearizer_order+1) :
        #         print(f"{ANSI_colors[current_color]}Ar_w{iii+2}{Ar[0:5, iii]}{clear}")
        #     else:
        #         print(f"{orange}Ar_c1{Ar[0:5, iii]}{clear}")
        # print('Ar_matrix',np.shape(Ar))
        return Ar

    def predictor(self, X_ref, V, w, bk):
        # Predict the output based on the input and weights
        # ---------- Delay reference signal and distorted signal to compute the SNR -----------------
        if self.causal_system:
            delay = int(self.linearizer_order / 2)
        else:
            delay = 0

        X_hat = np.zeros(np.shape(X_ref))
        for pos in range(0, len(X_ref)):
            v = np.expand_dims(V[pos], axis=0)
            Ar = self._make_matrix(v, bk)
            # print('delay:', delay)
            v_delayed=self._delay_sequence(v, delay)
            # -------------------------------------------------------------
            MM = Ar @ w
  # column vector
            X_hat[pos] = ActivationFunctions._quantize(v_delayed+MM.T, self.output_quantizer_resolution)  # +extra_bias vc_delayed+
            # X_hat[pos] = MM.T    #v_delayed+

        X_ref_delayed = self._delay_sequence(X_ref, delay) #int(self.linearizer_order / 2)

        # ---------- Filtering (static filter at the end of the linearizer)-----------------------------
        if self.static_filter is not None:
            # print('X_hat', np.shape(X_hat))
            X_hat = FIRFilter.filter_signal(X_hat, self.static_filter)
            trans_samp = max(delay, len(self.static_filter))
        else:
            trans_samp = delay
        # ---------- Remove transient effect -----------------------------------------
        if trans_samp > 0:
            X_ref_delayed = X_ref_delayed[:, trans_samp:-trans_samp]
            X_hat_tr = X_hat[:, trans_samp:-trans_samp]
        else:
            X_hat_tr = X_hat
        # ---------- SNDR and error ---------------------------------------------------
        SNR_value = SNR(X_ref_delayed, X_hat_tr, remove_DC=False, grouping_method='mean').result
        error = self._NMSE(X_ref_delayed, X_hat_tr, mode='min')
        return X_hat, error, SNR_value

    @staticmethod
    def _NMSE(y_true, y_pred, mode='min'):
        # Helper function to calculate the mean root square error (NMSE) between the true signal and predicted signal

        M = np.power(y_true - y_pred, 2)
        L = len(M[0, :])
        R = len(M[:, 0])
        c_vec = np.sum(M, axis=1)
        NMSE = np.sum(c_vec, axis=0)/L/R
        ###############
        # MMSE = np.max(M)/L/R
        return NMSE

    def _wk_solver(self, current_BK, continue_flag=False):
        rem_samp = 200
        A = []
        b = []
        R = len(self.V)
        for pos in range(0, R):
            v = np.expand_dims(self.V[pos], axis=0)
            if self.causal_system:
                delay = int(self.linearizer_order / 2)
            else:
                delay = 0
            v_delayed = self._delay_sequence(v, delay)
            Ar0 = self._make_matrix(v, current_BK)
            # -------------------------------------------------------------
            x_ref = np.expand_dims(self.X_ref[pos], axis=0)
            x_ref_delayed = self._delay_sequence(x_ref,delay)
            br0 = (x_ref_delayed - v_delayed).T
            if pos == 0:
                Ar = Ar0
                br = br0
            else:
                Ar = np.concatenate([Ar, Ar0], axis=0)
                br = np.concatenate([br, br0], axis=0)

            L = len(Ar[:])
            # print('Ar' + str(pos) + ':', np.shape(Ar), 'br' + str(pos) + ':', np.shape(br), 'L:', L, 'R:', R)

        A = Ar.T @ Ar
        A += self.L2_constant/R * np.identity(len(A))
        if np.linalg.matrix_rank(A) < len(A):   #####################
            wk = None
            print('Matrix A is singular')
            continue_flag = True
        else:
            b = Ar.T @ br
            wk = np.linalg.inv(A) @ b
            # print('A:', np.shape(A), 'b:', np.shape(b), 'L:', L, 'R:', R, 'rank:', np.linalg.matrix_rank(A))
        return wk, continue_flag

    def train(self):
        # Train the network by finding the best weights and biases
        best_wk = None
        best_BK = None
        best_L2 = None
        best_SNR = -100
        if self.X_ref.ndim == 1:
            self.X_ref = np.expand_dims(self.X_ref, axis=0)
            self.V = np.expand_dims(self.V, axis=0)
        # current_bk = np.zeros((self.branch_number, self.linearizer_order + 1))
        if self.reuse_bias:
            for seed in range(self.number_simulations):
                if self.activation_function == ActivationFunctions.polynomial and seed > 0:
                    # print('polynomial:', 'ak:', current_wk, 'bk:', current_bk)
                    continue
                current_bk = self._bkgenerator(seed, mode='linesearch')
                current_wk, continue_flag = self._wk_solver(current_bk)
                if continue_flag:
                    continue

                # if self.activation_function == ActivationFunctions.polynomial and seed ==0:
                #     print('polynomial:', 'ak:', current_wk, 'bk:', current_bk)
                current_x_hat, current_error, current_SNR_value = self.predictor(self.X_ref, self.V, current_wk, current_bk)
                # print(f'train{seed} NMSE:{current_error:.2e} SNDR:{current_SNR_value:.2f}')

                color_idx = 4
                if self.best_error > current_error:
                # if best_SNR < current_SNR_value:
                    best_wk = np.copy(current_wk)
                    best_BK = np.copy(current_bk)
                    self.best_error = np.copy(current_error)
                    best_SNR = np.copy(current_SNR_value)
                    best_L2 = self.L2_constant
                    color_idx = 0
                    print('best_L2:', best_L2, 'b_max:', np.max(best_BK), 'b_min:', np.min(best_BK))

                if color_idx == 0:
                    print(
                        f'{ANSI_colors[color_idx]}train{seed} L2:{self.L2_constant:.4e} NMSE:{current_error:.4e}/{self.best_error:.4e} SNDR:{current_SNR_value:.4f}/{best_SNR:.4f}{clear}')
        return best_wk, best_BK, best_L2, self.best_error, best_SNR

    def nested_loops(self, number_simulations, linearizer_order):
        # Generate combinations of nested loops for different bias values per block
        def generate(seed_indices, order):
            if order == linearizer_order:
                yield tuple(seed_indices)
                return
            for seed in range(number_simulations):
                yield from generate(seed_indices + [seed], order + 1)
        return generate([], 0)

class ActivationFunctions:
    """Activation Functions for Neural Networks.

       Methods:
       - _quantize(signals, internal_quantizer_resolution_bits=50): Quantizes input signals.
       - linear(x): Linear activation function.
       - ReLU(x): Rectified Linear Unit (ReLU) activation function.
       - ABS(x): Absolute value activation function.
       - polynomial(x): Polynomial activation function.

       Usage:
       activation_instance = ActivationFunctions()
       quantized_signals = activation_instance._quantize(signals)
       linear_result = activation_instance.linear(x)
       relu_result = activation_instance.ReLU(x)
       abs_result = activation_instance.ABS(x)
       poly_result = activation_instance.polynomial(x)
       """
    @classmethod
    def _quantize(cls, signals, internal_quantizer_resolution_bits=50):
        """Quantize input signals.

        Parameters:
        - signals (numpy.ndarray): Input signals.
        - internal_quantizer_resolution_bits (int, optional): Resolution of the internal quantizer. Default is 50.

        Returns:
        - numpy.ndarray: Quantized signals.
        """
        maximum_amplitude = np.max(np.abs(signals))
        quantization_step = 2 * maximum_amplitude / (2 ** internal_quantizer_resolution_bits - 1)
        quantized_signals = np.round(signals / quantization_step) * quantization_step

        return quantized_signals

    @staticmethod
    def linear(x):
        """
        Linear activation function.
        """
        x = np.copy(x)
        return x

    @staticmethod
    def ReLU(x):
        """
        ReLU's activation function.
        """
        x = np.copy(x)
        return np.maximum(0, x)

    @staticmethod
    def ABS(x):
        x = x.copy()
        """
        Absolute value activation function.
        """
        return abs(x)

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        """Hyperbolic tangent activation function."""
        return np.tanh(x)

    @staticmethod
    def softplus(x):
        """Smooth approximation of ReLU."""
        return np.log1p(np.exp(x)) # return np.log(1 + np.exp(x))

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """Leaky ReLU activation function."""
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def elu(x, alpha=1.0):
        """Exponential Linear Unit."""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def sign(x):
        """
        Sign activation function
        """
        x_copy = np.copy(x)
        x_copy[x_copy >= 0] = 1
        x_copy[x_copy < 0] = -1
        return x_copy.astype(int)

    @staticmethod
    def polynomial(vector, power, axis=0):
        """Polynomial activation function.
        Parameters:
        - x (numpy.ndarray): Input signals.
        - exponential (int, optional):
        Returns:
        - numpy.ndarray: Result of the polynomial activation function.
        """
        # x = np.copy(vector)
        # if axis == 0:
        #     x = x.T
        # for i in range(0, len(x)):
        #     x_input = np.copy(x[i])
        #     for ii in range(0, power+1):
        #         # x[i] = cls._quantize(x[i] * x_input)
        #         if ii == 0:
        #             x[i] =np.ones_like(x[i])
        #         elif ii == 1:
        #             x[i] =x_input
        #         else:
        #             x[i] = np.dot(x[i],x_input)

        # straightforward implementation
        x = np.copy(vector ** power)
        return x.T if axis == 0 else x
class SNR:
    """Signal-to-Noise Ratio (SNR) calculation class.

     Parameters:
     - X (numpy.ndarray): Original signal(s).
     - X_est (numpy.ndarray): Estimated signal(s).
     - remove_DC (bool, optional): Whether to remove the DC component. Default is True.
     - grouping_method (str, optional): Grouping method for multiple signals.
         Options: 'min', 'max', 'mean'. Default is 'mean'.

     Raises:
     - ValueError: If input dimensions are not supported.

     Attributes:
     - remove_DC (bool): Whether the DC component is removed.
     - X (numpy.ndarray): Original signal(s).
     - X_est (numpy.ndarray): Estimated signal(s).
     - grouping_method (str): Grouping method for multiple signals.
     - result: SNR result.

     Methods:
     - _simple_signal(X, X_est): Calculate SNR for a single signal.
     - _multiple_signal(X, X_est): Calculate SNR for multiple signals.

     Usage:
     snr_instance = SNR(X, X_est)
     print(snr_instance.result)
     """

    def __init__(self, X, X_est, remove_DC=True, grouping_method='mean', delay=0):
        """Initialize the SNR class."""
        self.remove_DC = remove_DC
        self.X = np.copy(X)
        self.X_est = np.copy(X_est)
        self.grouping_method = grouping_method
        self.delay = delay


        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
            X_est = np.expand_dims(X_est, axis=0)
            self.result = self._simple_signal(X, X_est)
        elif X.ndim == 2 and len(X) == 1:
            self.result = self._simple_signal(X, X_est)
        elif X.ndim == 2 and len(X) >= 1:
            self.result = self._multiple_signal(X, X_est)
        else:
            raise ValueError('Input dimensions not supported')

    def _simple_signal(self, X, X_est):
        """Calculate SNR for a single signal."""
        if self.remove_DC:
            X_est = X_est - np.mean(X_est)
            X_est = X_est - np.mean(X_est)
        num = np.sum(np.abs(X) ** 2)
        den = np.sum(np.abs(X_est - X) ** 2)
        SNR = 10 * np.log10(num / den)
        return SNR

    def _multiple_signal(self, X, X_est):
        """Calculate SNR for multiple signals."""
        SNR_est = []
        for position in range(0, len(X), 1):
            x = np.expand_dims(X[position], axis=0)
            x_est = np.expand_dims(X_est[position], axis=0)
            SNR_est.append(self._simple_signal(x, x_est))
        if self.grouping_method == 'min':
            result_SNR = np.min(SNR_est)
        elif self.grouping_method == 'max':
            result_SNR = np.max(SNR_est)
        elif self.grouping_method == 'mean':
            result_SNR = np.mean(SNR_est)
        else:
            result_SNR = SNR_est
        return result_SNR