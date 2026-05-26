#%%
# =================================================================
# ABS Linearizer — Matrix Inversion with |x| activation
# =================================================================
import numpy as np
from myfunctions.functions_helper import SpectrumAnalyzer, compute_statistics, storate_dictionary
from myclasses.linearizers_v19   import MatrixInversionLinearizer, ActivationFunctions
from myclasses.file_manipulation import PathManager, DataSetLoader

# ── ANSI colors ───────────────────────────────────────────────────────────────
green = '\033[0;32m'
clear = '\033[0m'

# =================================================================
# Configuration: only edit here
# =================================================================
ACTIVATION         = 'ReLU'   # 'ABS' | 'ReLU' | 'sigmoid' | 'tanh'| | 'softplus'| | 'sign'| | 'elu'|  'polynomial' # note: 'polynomial' implements Hammerstein model (the bias are set to 0)
BRANCH_NUMBER      = 8        # number of nonlinear branches
LINEARIZER_ORDER   = 2        # filter order (FIR tap length = order + 1)
L2                 = 1e-4     # L2 regularization strength (fixed)
NUMBER_SIMULATIONS = 10       # Monte-Carlo repetitions inside the linearizer
CAUSAL_SYSTEM      = True     # True → causal (introduces delay), False → non-causal
SAVE_RESULTS       = False    # set True to persist the trained model to disk
PLOTTING           = True     # set True to plot the spectrum

_ACTIVATION_MAP = {
    'ABS'        : ActivationFunctions.ABS,
    'ReLU'       : ActivationFunctions.ReLU,
    'polynomial' : ActivationFunctions.polynomial,
}

# Dataset parameters (must match the folder in datasets/)
ACTIVE_CARRIERS    = 31
DISTORTION_ORDER   = 2
DISTORTION_BRANCHES = 9

np.random.seed(1)

# =================================================================
# Load Data
# =================================================================
dataset_version = 210000 + 100*int(ACTIVE_CARRIERS) + 10*int(DISTORTION_ORDER) + int(DISTORTION_BRANCHES)
folder_prefix   = 'v' + str(dataset_version)

current_path, _ = PathManager().check_path_by_host()
train_path, test_path = PathManager().load_path(
    root_path=current_path, local_path='datasets',
    folder_prefix=folder_prefix, state='data'
)
simulation_path = PathManager().make_path(
    root_path=current_path, local_path='trained_model',
    current_path=folder_prefix, state='simulations'
)

loader_train = DataSetLoader(path=train_path, num_files=50)
data_train   = loader_train.load_dataset()
X      = data_train['pure_signal']
V      = data_train['distorted_signal']

loader_test = DataSetLoader(path=test_path, num_files=50)
data_test   = loader_test.load_dataset()
X_test = data_test['pure_signal']
V_test = data_test['distorted_signal']

# =================================================================
# Train
# =================================================================
linearizer = MatrixInversionLinearizer(
    X, V,
    BRANCH_NUMBER,
    [_ACTIVATION_MAP[ACTIVATION]],
    NUMBER_SIMULATIONS,
    L2,
    LINEARIZER_ORDER,
    causal_system=CAUSAL_SYSTEM,
    model_name='MI',
    best_error=np.inf,
    reuse_bias=True,
    static_filter=None,
)
wk, bk, _, error, snr = linearizer.train()
print(green + '-' * 10, 'model trained', '-' * 10 + clear)
print(f'  training SNR: {snr}   L2: {L2}')

# =================================================================
# Predict
# =================================================================
delay = int(LINEARIZER_ORDER / 2) if CAUSAL_SYSTEM else 0

X_hat,      error_train, _ = linearizer.predictor(X,      V,      wk, bk)
X_hat_test, error_test,  _ = linearizer.predictor(X_test, V_test, wk, bk)

X_delayed       = MatrixInversionLinearizer._delay_sequence(X,      delay)
V_delayed       = MatrixInversionLinearizer._delay_sequence(V,      delay)
X_test_delayed  = MatrixInversionLinearizer._delay_sequence(X_test, delay)
V_test_delayed  = MatrixInversionLinearizer._delay_sequence(V_test, delay)

# =================================================================
# Statistics
# =================================================================
DATA_BITS = 12
EST_BITS  = 14

print('-' * 20 + ' Train ' + '-' * 20)
SNR_X, SNR_V, SNR_X_hat = compute_statistics(
    X_delayed, V_delayed, X_hat,
    Q1_bits=DATA_BITS, Q2_bits=DATA_BITS, Q3_bits=EST_BITS,
    print_on_screen=True,
)
print('-' * 20 + ' Test  ' + '-' * 20)
SNR_X_test, SNR_V_test, SNR_X_hat_test = compute_statistics(
    X_test_delayed, V_test_delayed, X_hat_test,
    Q1_bits=DATA_BITS, Q2_bits=DATA_BITS, Q3_bits=EST_BITS,
    print_on_screen=True,
)

print(f"\nX:    [{np.min(X):.2f}, {np.max(X):.2f}]")
print(f"V:    [{np.min(V):.2f}, {np.max(V):.2f}]")
print(f"X̂:    [{np.min(X_hat):.2f}, {np.max(X_hat):.2f}]")
print(f"wk max: {np.max(abs(wk)):.4f}   L2: {L2}   bk max: {np.max(abs(bk)):.4f}")

# =================================================================
# Save (optional)
# =================================================================
if SAVE_RESULTS:
    base_filename = f"{simulation_path}/{folder_prefix}_ord_{LINEARIZER_ORDER}_ABS"
    results = {
        'branch_number'        : BRANCH_NUMBER,
        'wk'                   : wk,
        'bk'                   : bk,
        'L2'                   : L2,
        'SNR_X'                : SNR_X,
        'SNR_V'                : SNR_V,
        'SNR_X_hat'            : SNR_X_hat,
        'SNR_X_hat_test'       : SNR_X_hat_test,
        'error_train'          : error_train,
        'error_test'           : error_test,
    }
    storate_dictionary(base_filename, results)
    print(f'Results saved to {base_filename}.pickle')
else:
    print('Results not saved (SAVE_RESULTS=False)')

# =================================================================
# Plot (optional)
# =================================================================
if PLOTTING:
    title = ['Distorted signal', 'Linearized signal']
    signal_idx = 0
    SpectrumAnalyzer().plot_frequency_domain(
        [np.expand_dims(V_test[signal_idx], 0), np.expand_dims(X_hat_test[signal_idx], 0)],
        title,
        window_type='Blackmanharris',
        save_path=f"{simulation_path}/{folder_prefix}_{signal_idx}",
        save_fig=False,
    )
