"""Implements several channel estimation baselines"""

from typing import Tuple, List
import numpy as np
import tensorflow as tf
from copy import deepcopy
from cebed.utils import tfinterpolate
from typing import List, Dict
import numpy as np
from sionna.phy.ofdm import LSChannelEstimator, LMMSEInterpolator
from cebed.utils import unflatten_last_dim
from cebed.utils_eval import mse, estimate_covariance_matrices
from cebed.envs import OfdmEnv

from sionna.phy.ofdm import LinearDetector, KBestDetector, EPDetector, MMSEPICDetector

# ---------------------------------------------------------------------------- #
#                              Evaluate Baselines                              #
# ---------------------------------------------------------------------------- #
def evaluate_baselines(
    obj,
    noisy_signals: tf.Tensor,
    noiseless_signals: tf.Tensor,
    channels: tf.Tensor,
    snr: int,
    baselines: List[str],
) -> Dict[str, float]:
    
    """Evaluates a list of baselines"""

    results = {}
    noise_lin = tf.pow(10.0, -snr / 10.0)

    for baseline in baselines:
        if baseline == "LS":
            h_ls_lin = obj.evaluate_ls(noisy_signals)
            lin_mse = mse(tf.squeeze(channels), tf.squeeze(h_ls_lin))
            results["LS"] = lin_mse.numpy()
            
        elif baseline == "LMMSE":
            h_ls = obj.evaluate_ls(noisy_signals, interpolate=False)
            h_lmmse = obj.evaluate_lmmse(noiseless_signals, channels, h_ls, snr)
            lmmse_mse = mse(tf.squeeze(channels), tf.squeeze(h_lmmse))
            results["LMMSE"] = lmmse_mse.numpy()

        elif baseline == "ALMMSE":
            if isinstance(obj.dataset.env, OfdmEnv):
                pass
            almmse_h_hat = obj.evaluate_almmse(noisy_signals, noise_lin)
            almmse_mse = mse(tf.squeeze(channels), tf.squeeze(almmse_h_hat))
            results["ALMMSE"] = almmse_mse.numpy()
        
        elif baseline == "DDCE":
            h_ddce, converge_info = obj.evaluate_ddce(noisy_signals, snr)

            channels = tf.cast(channels, tf.complex64)
            h_ddce = tf.cast(h_ddce, tf.complex64)
            ddce_mse = mse(tf.squeeze(channels), tf.squeeze(h_ddce))
            results["DDCE"] = ddce_mse.numpy()
            # results["DDCE_converge_iter"] = len(converge_info)
            
        else:
            raise ValueError(f"Baseline is not supported {baseline}")

    return results

def evaluate_ls(
    obj, noisy_signals: tf.Tensor, interpolate: bool = True
) -> tf.Tensor:
    """LS method with bilinear interpolation"""

    y_pilot_noisy = obj.dataset.env.extract_at_pilot_locations(noisy_signals)
    # [batch, num_r, num_r_ants, n_t, n_t_streams, num_pilots]
    h_ls = tf.math.divide_no_nan(y_pilot_noisy, obj.dataset.pilots)
    # [batch, num_r, num_r_ants, n_t, n_t_streams, num_pilot_symbols, num_pilot_subcarriers]
    h_ls = unflatten_last_dim(
        h_ls, (obj.dataset.num_pilot_symbols, obj.dataset.num_pilot_subcarriers)
    )

    if not interpolate:
        return h_ls
    h_ls_lin = linear_ls_baseline(
        h_ls, obj.dataset.num_symbols, obj.dataset.num_subcarries
    )

    return h_ls_lin

def evaluate_lmmse(
    obj,
    noiseless_signals: tf.Tensor,
    channels: tf.Tensor,
    hls: tf.Tensor,
    snr_db: int,
) -> tf.Tensor:
    
    """Ideal LMMSE: given noiseless LS channel estimates, then do lmmse interpolation (details as in lmmse_baseline(..)) """ 
    y_pilot_noise_free = obj.dataset.env.extract_at_pilot_locations(
        noiseless_signals
    )
    # Channels at pilot positions noise free
    noiseless_hls = tf.math.divide_no_nan(y_pilot_noise_free, obj.dataset.pilots)
    # [batch, num_r, num_r_ants, n_t, n_t_streams, num_pilot_symbols, num_pilot_subcarriers]
    noiseless_hls = unflatten_last_dim(
        noiseless_hls,
        (obj.dataset.num_pilot_symbols, obj.dataset.num_pilot_subcarriers),
    )
    h_lmmse = lmmse_baseline(
        noiseless_hls, 
        channels,
        hls,
        snr_db,
        obj.dataset.env.pilot_ofdm_symbol_indices,
        obj.dataset.num_symbols,
        obj.dataset.num_subcarries,
    )

    return h_lmmse


def evaluate_almmse(obj, noisy_signals: tf.Tensor, noise_lin: float) -> tf.Tensor:
    """ALMMSE baseline. Only work with Sionna environments
    Given noisy LS estimate, then do LMMSE interpolation with Monte-Carlo covariance estimation.
    """

    # estimate the covariance matrices
    freq_cov_mat, time_cov_mat, space_cov_mat = estimate_covariance_matrices(
        obj.dataset.env, save_dir=f"{obj.log_dir}/cov_mats"
    ) 
    order = "f-t"

    if obj.dataset.env.config.num_rx_antennas > 1:
        order = "f-t-s"
    lmmse_int = LMMSEInterpolator(
        obj.dataset.env.rg.pilot_pattern,
        time_cov_mat,
        freq_cov_mat,
        space_cov_mat,
        order=order,
    )
    lmmse_estimator = LSChannelEstimator(
        obj.dataset.env.rg, interpolator=lmmse_int
    )

    almmse_h_hat, _ = lmmse_estimator(noisy_signals, noise_lin)

    return almmse_h_hat


def evaluate_ddce(obj, noisy_signals: tf.Tensor, snr_db: int, 
                  iter_times = 100,
                  converge_threshold = 1e-5,
                  max_iterations = 100) -> tf.Tensor:
    
    # get the intial LS estimate
    h_ls_lin = obj.evaluate_ls(noisy_signals)
    # check the shape of h_ls_lin
    print(f"h_ls_lin shape: {h_ls_lin.shape}")
    
    converge_info = []
        # Step 1: Iterative DDCE process
    H_current = deepcopy(h_ls_lin)
    
    print(f"Starting DDCE with {iter_times} iterations...")
    
    for iteration in range(iter_times):
        
        H_previous = deepcopy(H_current)

        # Equalize and make hard decisions
        H_safe = np.where(np.abs(H_current) < 1e-10, 1e-10, H_current)

        # equalize the currect signal
        equalized_symbols = noisy_signals / H_safe # TODO: can further use LMMSE detector

        # hard decision
        hard_decisions = qam4_hard_decision(equalized_symbols)
        
        # Re-estimate channel
        decisions_safe = np.where(np.abs(hard_decisions) < 1e-10, 1e-10, hard_decisions)
        H_current = noisy_signals / decisions_safe 
         
        channel_change = np.mean(np.abs(H_current - H_previous)**2)
        converge_info.append(channel_change)

        if iteration % 10 == 0:
            print(f"DDCE iteration {iteration}")
            print(f"Channel change: {channel_change:.4f}")    

        if channel_change < converge_threshold:
            print(f"DDCE iteration {iteration}")
            print(f"Channel change: {channel_change:.4f}")
            break
        
        
    return H_current, converge_info


# ---------------------------------------------------------------------------- #
def qam4_hard_decision(symbols: np.ndarray) -> np.ndarray:
    """qam4 hard decision mapping - same as before"""
    hard_decisions = np.zeros_like(symbols, dtype=complex)
    angles = np.angle(symbols)
    
    # Map to qam4 constellation based on phase
    mask1 = (angles > 0) & (angles < np.pi/2)
    hard_decisions[mask1] = (1 + 1j) / np.sqrt(2)  # Normalize to unit magnitude
    
    mask2 = (angles > np.pi/2) & (angles < np.pi)
    hard_decisions[mask2] = (-1 + 1j) / np.sqrt(2)  # Normalize to unit magnitude
    
    mask3 = (angles > -np.pi) & (angles < -np.pi/2)
    hard_decisions[mask3] = (-1 - 1j) / np.sqrt(2)  # Normalize to unit magnitude
    
    mask4 = (angles > -np.pi/2) & (angles < 0)
    hard_decisions[mask4] = (1 - 1j) / np.sqrt(2)  # Normalize to unit magnitude
    return hard_decisions

# ---------------------------------------------------------------------------- #
def linear_interpolation(
    mat, new_shape: Tuple[int], method: str = "bilinear"
) -> tf.Tensor:
    """
    Upscale a matrice to a given shape using linear interpolation
    For complex inputs, the real and imaginary parts are interpolated seperatly
    """
    mat_interpolated = tfinterpolate(mat, new_shape, method=method)

    return mat_interpolated


def lmmse_interpolation(h_p, h, h_ls, snr_db, pilot_locations):
    """
    LMMSE interpolation
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4699619
    """

    shape = tf.shape(h_ls)
    n_rx_ant = shape[2]
    n_tx = shape[3]
    h_p = tf.reshape(h_p, (shape[0], n_rx_ant, n_tx, shape[-2], shape[-1]))
    h_ls = tf.reshape(h_ls, (shape[0], n_rx_ant, n_tx, shape[-2], shape[-1]))
    h = tf.reshape(h, (shape[0], n_rx_ant, n_tx, h.shape[-2], h.shape[-1]))

    # [batch, nr, nt, n_pilot_symbols, n_subcarrriers]
    h_mmse = np.zeros((shape[0], n_rx_ant, n_tx, shape[-2], h.shape[-1])).astype(
        complex
    )

    for b in range(shape[0]):
        for nt in range(n_tx):
            for nr in range(n_rx_ant):
                for k, i in enumerate(pilot_locations):
                    hp = h_p[b, nr, nt, k, :].numpy()[:, None]
                    hi = h[b, nr, nt, i, :].numpy()[:, None]
                    rhp = np.matmul(hp, hp.conj().T)
                    rhhp = np.matmul(hi, hp.conj().T)
                    pinv = np.linalg.pinv(
                        rhp + (1 / (10.0 ** (snr_db / 10.0))) * np.eye(rhp.shape[0])
                    )
                    a = np.matmul(rhhp, pinv) # a is the MMSE filter matrix
                    h_mmse[b, nr, nt, k, :] = np.matmul(a, h_ls[b, nr, nt, k, :])

    return h_mmse


def linear_ls_baseline(
    h_ls: tf.Tensor, num_ofdm_symbols: int, num_ofdm_subcarriers: int
):
    """Linear interpolation of the LS estimates"""
    # [batch, num_r, num_r_ants, n_t, n_t_streams, num_pilot_symbols, num_pilot_subcarriers]

    shape = tf.shape(h_ls)
    n_rx_ant = shape[2]
    n_tx = shape[3]
    # [num_batch, nrx_ant*nt, num_pilot_symbols,num_pilot_subcarriers]
    h_ls_r = tf.reshape(h_ls, (shape[0], n_rx_ant * n_tx, shape[-2], shape[-1]))
    # [num_batch, num_pilot_symbols,num_pilot_subcarriers, nrx_ant*nt]
    h_ls_r = tf.transpose(h_ls_r, [0, 2, 3, 1])
    # [num_batch, num_symbols,num_subcarriers, nrx_ant*nt]
    h_ls_lin = linear_interpolation(h_ls_r, (num_ofdm_symbols, num_ofdm_subcarriers))
    # [num_batch, , nrx_ant*nt, num_pilot_symbols,num_pilot_subcarriers]
    h_ls_lin = tf.transpose(h_ls_lin, [0, 3, 1, 2])
    # [num_batch, , nrx_ant, nt, num_pilot_symbols,num_pilot_subcarriers]
    shape = tf.shape(h_ls_lin)
    h_ls_lin = tf.reshape(h_ls_lin, (shape[0], n_rx_ant, n_tx, shape[-2], shape[-1]))

    return h_ls_lin


def lmmse_baseline(
    h_p: tf.Tensor,
    h_freq: tf.Tensor,
    h_ls: tf.Tensor,
    snr_db: int,
    pilot_locations: List[int],
    num_ofdm_symbols: int,
    num_ofdm_subcarriers: int,
) -> tf.Tensor:
    """
    Idea LMMSE baseline.
    :param h_p: The LS estimates at pilot positions of noise-free channel
    :param h_freq: The true channel coefficients
    :param h_ls: The LS estimates at pilot positions of noisy channel
    :param snr_db: The noise level in dB
    :param pilot_locations: A list of the pilot symbol indices
    :param num_ofdm_symbols: The number of symbols in the resource grid
    :param num_ofdm_subcarriers: The number of subcarriers in the resource grid

    :return The LMMSE estimates
    
    The process seems tricky:
    1. LMMSE interpolation to noise-free LS estimates, output dimension [..., num_pilot_symbols,num_pilot_subcarriers]
    2. Linear interpolation to output dimension [...,  num_symbols,num_subcarriers, ...]

    """

    # [num_batch, nrx_ant, nt, num_pilot_symbols,num_pilot_subcarriers]
    h_lmmse = lmmse_interpolation(h_p, h_freq, h_ls, snr_db, pilot_locations) # lmmse interpolation to noise-free estimates

    shape = tf.shape(h_lmmse)
    n_rx_ant = shape[1]
    n_tx = shape[2]
    # [num_batch, nrx_ant*nt, num_pilot_symbols,num_pilot_subcarriers]
    h_lmmse_r = tf.reshape(h_lmmse, (shape[0], n_rx_ant * n_tx, shape[-2], shape[-1]))
    # [num_batch, num_pilot_symbols,num_pilot_subcarriers, nrx_ant*nt]
    h_lmmse_r = tf.transpose(h_lmmse_r, [0, 2, 3, 1])
    # [num_batch, num_symbols,num_subcarriers, nrx_ant*nt]
    h_lmmse_frame = linear_interpolation(
        h_lmmse_r, (num_ofdm_symbols, num_ofdm_subcarriers)
    )
    # [num_batch, , nrx_ant*nt, num_pilot_symbols,num_pilot_subcarriers]
    h_lmmse_frame = tf.transpose(h_lmmse_frame, [0, 3, 1, 2])
    # [num_batch, , nrx_ant, nt, num_pilot_symbols,num_pilot_subcarriers]
    shape = tf.shape(h_lmmse_frame)
    h_lmmse_frame = tf.reshape(
        h_lmmse_frame, (shape[0], n_rx_ant, n_tx, shape[-2], shape[-1])
    )

    return h_lmmse_frame
