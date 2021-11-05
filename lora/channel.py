""" LPWAN Simulator: Simulate the propagation channel between the devices in the wireless network     
============================================
Utilities (:mod:`lora.channel`)
============================================
.. autosummary::
   :toctree: generated/   
   awgn                 -- Addditive White Gaussian Noise (AWGN) Channel.
   simpleRayleigh       -- Simple Rayleigh fading Channel.
   
"""

# Import Library
# import numpy as np

from numpy import abs, sqrt, linspace
from numpy.random import randn, normal
import math



def awgn(input_signal, snr_dB, rate=1.0):
    """
    Addditive White Gaussian Noise (AWGN) Channel.
    Parameters
    ----------
    input_signal : 1D ndarray of floats
        Input signal to the channel.
    snr_dB : float
        Output SNR required in dB.
    rate : float
        Rate of the a FEC code used if any, otherwise 1.
    Returns
    -------
    output_signal : 1D ndarray of floats
        Output signal from the channel with the specified SNR.
    """

    avg_energy = sum(abs(input_signal) * abs(input_signal)) / len(input_signal)
    snr_linear = 10 ** (snr_dB / 10.0)
    noise_variance = avg_energy / (2 * rate * snr_linear)
    if isinstance(input_signal[0], complex):
        noise = (sqrt(noise_variance) * randn(len(input_signal))) + (
                sqrt(noise_variance) * randn(len(input_signal)) * 1j)
    else:
        noise = sqrt(2 * noise_variance) * randn(len(input_signal))

    output_signal = input_signal + noise

    return output_signal


def simpleRayleigh(input_signal, snr_dB, rate=1.0):
    """
    Simple Rayleigh fading Channel.
    Parameters
    ----------
    input_signal : 1D ndarray of floats
        Input signal to the channel.
    snr_dB : float
        Output SNR required in dB.
    rate : float
        Rate of the a FEC code used if any, otherwise 1.
    Returns
    -------
    output_signal : 1D ndarray of floats
        Output signal from the channel with the specified SNR.
    """

    avg_energy = sum(abs(input_signal) * abs(input_signal)) / len(input_signal)
    snr_linear = 10 ** (snr_dB / 10.0)
    noise_variance = avg_energy / (2 * rate * snr_linear)
    ch_coeff = sqrt(normal(0, 1) ** 2 + normal(0, 1) ** 2) / sqrt(2)

    if isinstance(input_signal[0], complex):
        noise = (sqrt(noise_variance) * randn(len(input_signal))) + (
                sqrt(noise_variance) * randn(len(input_signal)) * 1j)
    else:
        noise = sqrt(2 * noise_variance) * randn(len(input_signal))

    output_signal = input_signal * ch_coeff + noise

    return output_signal


def nakagami(input_signal, mu, Omega, size, rg):

    h1 = math.sqrt(rg.gamma(mu, Omega / mu, size))
    output_signal = h1 * input_signal

    return output_signal






