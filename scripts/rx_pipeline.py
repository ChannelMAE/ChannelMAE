
import matplotlib.pyplot as plt
import numpy as np
import pickle

import tensorflow as tf

from sionna.mimo import StreamManagement
from sionna.utils import QAMSource, compute_ser, BinarySource, sim_ber, ebnodb2no, QAMSource
from sionna.mapping import Mapper
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEInterpolator, LinearDetector, KBestDetector, EPDetector, MMSEPICDetector
from sionna.channel import GenerateOFDMChannel, OFDMChannel, gen_single_sector_topology
from sionna.channel.tr38901 import UMi, Antenna, PanelArray
from sionna.fec.ldpc import LDPC5GEncoder
from sionna.fec.ldpc import LDPC5GDecoder


# set all the parameters
NUM_OFDM_SYMBOLS = 14
FFT_SIZE = 2048
SUBCARRIER_SPACING = 15e3
CHANNEL_MODEL = GenerateOFDMChannel(UMi, Antenna(PanelArray(4, 4)), fft_size=FFT_SIZE, subcarrier_spacing=SUBCARRIER_SPACING)
FREQ_COV_MAT = np.load("data/freq_cov_mat.npy")
TIME_COV_MAT = np.load("data/time_cov_mat.npy")
SPACE_COV_MAT = np.load("data/space_cov_mat.npy")
SPEED = 3.0e8/4.0



class MIMOOFDMLink(tf.keras.Model):

    def __init__(self, output, det_method, perf_csi, num_tx, num_bits_per_symbol, det_param=None, coderate=0.5, **kwargs):
        super().__init__(kwargs)

        assert det_method in ('lmmse', 'k-best', 'ep', 'mmse-pic'), "Unknown detection method"

        self._output = output
        self.num_tx = num_tx
        self.num_bits_per_symbol = num_bits_per_symbol
        self.coderate = coderate
        self.det_method = det_method
        self.perf_csi = perf_csi

        # Configure the resource grid
        rg = ResourceGrid(num_ofdm_symbols=NUM_OFDM_SYMBOLS,
                          fft_size=FFT_SIZE,
                          subcarrier_spacing=SUBCARRIER_SPACING,
                          num_tx=num_tx,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2,11])
        self.rg = rg

        # Stream management
        sm = StreamManagement(np.ones([1,num_tx], int), 1)

        # Codeword length and number of information bits per codeword
        n = int(rg.num_data_symbols*num_bits_per_symbol)
        k = int(coderate*n)
        self.n = n
        self.k = k

        # If output is symbol, then no FEC is used and hard decision are output
        hard_out = (output == "symbol")
        coded = (output == "bit")
        self.hard_out = hard_out
        self.coded = coded

        ##################################
        # Transmitter
        ##################################
        self.binary_source = BinarySource()
        self.mapper = Mapper(constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, return_indices=True)
        self.rg_mapper = ResourceGridMapper(rg)
        if coded:
            self.encoder = LDPC5GEncoder(k, n, num_bits_per_symbol=num_bits_per_symbol)

        ##################################
        # Channel
        ##################################
        self.channel = OFDMChannel(CHANNEL_MODEL, rg, return_channel=True)

        ###################################
        # Receiver
        ###################################

        # Channel estimation
        if not self.perf_csi:
            freq_cov_mat = tf.constant(FREQ_COV_MAT, tf.complex64)
            time_cov_mat = tf.constant(TIME_COV_MAT, tf.complex64)
            space_cov_mat = tf.constant(SPACE_COV_MAT, tf.complex64)
            lmmse_int_time_first = LMMSEInterpolator(rg.pilot_pattern, time_cov_mat, freq_cov_mat, space_cov_mat, order='t-f-s')
            self.channel_estimator = LSChannelEstimator(rg, interpolator=lmmse_int_time_first) # LMMSE interpolation with covariance...

        # Detection
        if det_method == "lmmse":
            self.detector = LinearDetector("lmmse", output, "app", rg, sm, constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, hard_out=hard_out)
        elif det_method == 'k-best':
            if det_param is None:
                k = 64
            else:
                k = det_param
            self.detector = KBestDetector(output, num_tx, k, rg, sm, constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, hard_out=hard_out)
        elif det_method == "ep":
            if det_param is None:
                l = 10
            else:
                l = det_param
            self.detector = EPDetector(output, rg, sm, num_bits_per_symbol, l=l, hard_out=hard_out)
        elif det_method == 'mmse-pic':
            if det_param is None:
                l = 4
            else:
                l = det_param
            self.detector = MMSEPICDetector(output, rg, sm, 'app', num_iter=l, constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, hard_out=hard_out)

        if coded:
            self.decoder = LDPC5GDecoder(self.encoder, hard_out=False)

    @tf.function
    def call(self, batch_size, ebno_db):


        ##################################
        # Transmitter
        ##################################

        if self.coded:
            b = self.binary_source([batch_size, self.num_tx, 1, self.k])
            c = self.encoder(b)
        else:
            c = self.binary_source([batch_size, self.num_tx, 1, self.n])
        bits_shape = tf.shape(c)
        x,x_ind = self.mapper(c)
        x_rg = self.rg_mapper(x)

        ##################################
        # Channel
        ##################################

        no = ebnodb2no(ebno_db, self.num_bits_per_symbol, self.coderate, resource_grid=self.rg)
        topology = gen_single_sector_topology(batch_size, self.num_tx, 'umi', min_ut_velocity=SPEED, max_ut_velocity=SPEED)
        CHANNEL_MODEL.set_topology(*topology)
        y_rg, h_freq = self.channel((x_rg, no))

        ###################################
        # Receiver
        ###################################

        # Channel estimation
        if self.perf_csi:
            h_hat = h_freq
            err_var = 0.0
        else:
            h_hat,err_var = self.channel_estimator((y_rg,no))

        # Detection
        if self.det_method == "mmse-pic":
            if self._output == "bit":
                prior_shape = bits_shape
            elif self._output == "symbol":
                prior_shape = tf.concat([tf.shape(x), [self.num_bits_per_symbol]], axis=0)
            prior = tf.zeros(prior_shape)
            det_out = self.detector((y_rg,h_hat,prior,err_var,no))
        else:
            det_out = self.detector((y_rg,h_hat,err_var,no))

        # (Decoding) and output
        if self._output == "bit":
            llr = tf.reshape(det_out, bits_shape)
            b_hat = self.decoder(llr)
            return b, b_hat
        elif self._output == "symbol":
            x_hat = tf.reshape(det_out, tf.shape(x_ind))
            # FIXME: return x instead of x_ind
            return x_ind, x_hat
        

def run_sim(num_tx, num_bits_per_symbol, output, ebno_dbs, perf_csi, det_param=None):

    lmmse = MIMOOFDMLink(output, "lmmse", perf_csi, num_tx, num_bits_per_symbol, det_param)
    k_best = MIMOOFDMLink(output, "k-best", perf_csi, num_tx, num_bits_per_symbol, det_param)
    ep = MIMOOFDMLink(output, "ep", perf_csi, num_tx, num_bits_per_symbol, det_param)
    mmse_pic = MIMOOFDMLink(output, "mmse-pic", perf_csi, num_tx, num_bits_per_symbol, det_param)

    if output == "symbol":
        soft_estimates = False # why soft_estimates = False????
        ylabel = "Uncoded SER"
    else:
        soft_estimates = True
        ylabel = "Coded BER"

    er_lmmse,_ = sim_ber(lmmse,
        ebno_dbs,
        batch_size=64,
        max_mc_iter=200,
        num_target_block_errors=200,
        soft_estimates=soft_estimates);

    er_ep,_ = sim_ber(ep,
        ebno_dbs,
        batch_size=64,
        max_mc_iter=200,
        num_target_block_errors=200,
        soft_estimates=soft_estimates);

    er_kbest,_ = sim_ber(k_best,
       ebno_dbs,
       batch_size=64,
       max_mc_iter=200,
       num_target_block_errors=200,
       soft_estimates=soft_estimates);

    er_mmse_pic,_ = sim_ber(mmse_pic,
       ebno_dbs,
       batch_size=64,
       max_mc_iter=200,
       num_target_block_errors=200,
       soft_estimates=soft_estimates);

    return er_lmmse, er_ep, er_kbest, er_mmse_pic

# Range of SNR (dB)
EBN0_DBs = np.linspace(-10., 20.0, 10)

# Number of transmitters
NUM_TX = 4

# Modulation order (number of bits per symbol)
NUM_BITS_PER_SYMBOL = 4 # 16-QAM

SER = {} # Store the results

# Perfect CSI
ser_lmmse, ser_ep, ser_kbest, ser_mmse_pic = run_sim(NUM_TX, NUM_BITS_PER_SYMBOL, "symbol", EBN0_DBs, True)
SER['Perf. CSI / LMMSE'] = ser_lmmse
SER['Perf. CSI / EP'] = ser_ep
SER['Perf. CSI / K-Best'] = ser_kbest
SER['Perf. CSI / MMSE-PIC'] = ser_mmse_pic

# Imperfect CSI
ser_lmmse, ser_ep, ser_kbest, ser_mmse_pic = run_sim(NUM_TX, NUM_BITS_PER_SYMBOL, "symbol", EBN0_DBs, False)
SER['Ch. Est. / LMMSE'] = ser_lmmse
SER['Ch. Est. / EP'] = ser_ep
SER['Ch. Est. / K-Best'] = ser_kbest
SER['Ch. Est. / MMSE-PIC'] = ser_mmse_pic


BER = {} # Store the results

# Perfect CSI
ber_lmmse, ber_ep, ber_kbest, ber_mmse_pic = run_sim(NUM_TX, NUM_BITS_PER_SYMBOL, "bit", EBN0_DBs, True)
BER['Perf. CSI / LMMSE'] = ber_lmmse
BER['Perf. CSI / EP'] = ber_ep
BER['Perf. CSI / K-Best'] = ber_kbest
BER['Perf. CSI / MMSE-PIC'] = ber_mmse_pic

# Imperfect CSI
ber_lmmse, ber_ep, ber_kbest, ber_mmse_pic = run_sim(NUM_TX, NUM_BITS_PER_SYMBOL, "bit", EBN0_DBs, False)
BER['Ch. Est. / LMMSE'] = ber_lmmse
BER['Ch. Est. / EP'] = ber_ep
BER['Ch. Est. / K-Best'] = ber_kbest
BER['Ch. Est. / MMSE-PIC'] = ber_mmse_pic