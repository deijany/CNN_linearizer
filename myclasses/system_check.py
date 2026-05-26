"""
system_check.py — Hardware and runtime environment summary.

Cross-platform: works on Google Colab (Linux + nvidia-smi + TPU),
local macOS (Metal GPU, sysctl), and generic Linux.
"""

import os
import subprocess

import tensorflow as tf
from psutil import virtual_memory

from myclasses.env_utils import get_cpu_info, is_colab


class SystemCheck:
    RED   = '\033[91m'
    GREEN = '\033[92m'
    RESET = '\033[0m'

    def __init__(self):
        self.tf_version = tf.__version__
        _mem = virtual_memory()
        self.ram_total_gib = _mem.total     / (1024 ** 3)  # total installed RAM
        self.ram_avail_gib = _mem.available / (1024 ** 3)  # actually free right now
        self.ram_gb = self.ram_total_gib                   # threshold check uses total

    def _check_gpu(self):
        """Check for GPU availability and print GPU info if available."""
        if tf.config.list_physical_devices('GPU'):
            print(f"{self.GREEN}A GPU is connected:{self.RESET}")
            try:
                # nvidia-smi works on Linux/Colab; on macOS Metal it doesn't exist
                gpu_info = subprocess.run(
                    ['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE
                ).stdout.decode('utf-8')
                print(gpu_info if gpu_info.strip() else '  (nvidia-smi returned no output)')
            except FileNotFoundError:
                print('  (Metal/Apple GPU — nvidia-smi not available on macOS)')
        else:
            print(f"{self.RED}No GPU connected.{self.RESET}")

    def _check_tpu(self):
        """Check for TPU availability — Colab-only, skipped gracefully elsewhere."""
        if is_colab() and 'COLAB_TPU_ADDR' in os.environ:
            try:
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
                tf.config.experimental_connect_to_cluster(tpu)
                tf.distribute.experimental.TPUStrategy(tpu)
                print(f'{self.GREEN}Running on TPU '
                      f'{tpu.cluster_spec().as_dict()["worker"]}{self.RESET}')
            except ValueError:
                print(f'{self.RED}ERROR: Not connected to a TPU runtime.{self.RESET}')
        else:
            print(f"{self.RED}No TPU connected.{self.RESET}")

    def _check_cpu(self):
        """Print CPU info — uses lscpu on Linux/Colab, sysctl on macOS."""
        print(f"{self.GREEN}CPU Info:{self.RESET}")
        print(get_cpu_info())

    def _check_ram(self):
        """Print RAM info."""
        if self.ram_gb < 20:
            print(f'{self.RED}Not using a high-RAM runtime{self.RESET}')
        else:
            print(f'{self.GREEN}You are using a high-RAM runtime!{self.RESET}')
        print(f'RAM: {self.ram_avail_gib:.1f} GiB available / {self.ram_total_gib:.1f} GiB total')

    def print_summary(self):
        """Print a summary of TF version, GPU, TPU, CPU, and RAM."""
        print('---' * 28)
        print(f'{self.GREEN}tensorflow_version={self.tf_version}{self.RESET}')
        print('---' * 28)
        self._check_gpu()
        self._check_tpu()
        self._check_cpu()
        self._check_ram()
        print('---' * 28)
