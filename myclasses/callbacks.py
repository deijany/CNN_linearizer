"""
callbacks.py — Custom Keras training callbacks.

Reusable across notebooks and scripts; no notebook-specific logic here.
"""

import sys

import tensorflow as tf


class UniversalPrintCallback(tf.keras.callbacks.Callback):
    """
    Print epoch loss on a single updating line in Jupyter, or newline-separated
    in plain terminals — works identically in Colab, JupyterLab, and scripts.
    """

    def __init__(self):
        super().__init__()
        self.header = ''  # set before each fit() call to keep a persistent status line

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        safe_logs = {}
        for k, v in logs.items():
            try:
                safe_logs[k] = float(v)
            except Exception:
                safe_logs[k] = v

        loss     = safe_logs.get('loss', 0.0)
        val_loss = safe_logs.get('val_loss', None)

        if val_loss is not None:
            message = (f"Epoch {epoch + 1}: "
                       f"Loss = {loss:.7f}, Validation Loss = {val_loss:.7f}")
        else:
            message = f"Epoch {epoch + 1}: Loss = {loss:.7f}, Validation Loss = N/A"

        if 'ipykernel' in sys.modules:
            from IPython.display import clear_output
            clear_output(wait=True)
            if self.header:
                print(self.header)
            print(message, flush=True)
        else:
            sys.stdout.write('\r' + message)
            sys.stdout.flush()
