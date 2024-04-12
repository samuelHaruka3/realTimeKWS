import sys
from threading import Thread

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QScrollArea
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import librosa.display
from matplotlib.figure import Figure
from collections import deque
from torch.utils.data import DataLoader
import os
import librosa
import numpy as np
import time
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms
from spikingjelly.activation_based import neuron, surrogate, monitor
from spikingjelly.datasets.speechcommands import SPEECHCOMMANDS
from spikingjelly.activation_based.functional import reset_net
from scipy.signal import savgol_filter
import math
import argparse
from typing import Optional
import pyaudio
import torch
from torchaudio.transforms import Spectrogram


label_dict = {'yes': 0, 'stop': 1, 'no': 2, 'right': 3, 'up': 4, 'left': 5, 'on': 6, 'down': 7, 'off': 8, 'go': 9,
              'bed': 10, 'three': 10, 'one': 10, 'four': 10, 'two': 10, 'five': 10, 'cat': 10, 'dog': 10, 'eight': 10,
              'bird': 10, 'happy': 10, 'sheila': 10, 'zero': 10, 'wow': 10, 'marvin': 10, 'house': 10, 'six': 10,
              'seven': 10, 'tree': 10, 'nine': 10, '_silence_': 11}
label_cnt = len(set(label_dict.values()))
n_mels = 40
f_max = 4000
f_min = 20
delta_order = 0
size = 16000
try:
    import cupy

    backend = 'cupy'
except ModuleNotFoundError:
    backend = 'torch'
    print('Cupy is not intalled. Using torch backend for neurons.')


def mel_to_hz(mels, dct_type):
    if dct_type == 'htk':
        return 700.0 * (10 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = math.log(6.4) / 27.0  # step size for log region

    if torch.is_tensor(mels) and mels.ndim:
        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * \
                       torch.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * math.exp(logstep * (mels - min_log_mel))

    return freqs


def hz_to_mel(frequencies, dct_type):
    if dct_type == 'htk':
        if torch.is_tensor(frequencies) and frequencies.ndim:
            return 2595.0 * torch.log10(1.0 + frequencies / 700.0)
        return 2595.0 * math.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = math.log(6.4) / 27.0  # step size for log region

    if torch.is_tensor(frequencies) and frequencies.ndim:
        # If we have array data, vectorize
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + \
                      torch.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + math.log(frequencies / min_log_hz) / logstep

    return mels


def create_fb_matrix(
        n_freqs: int,
        f_min: float,
        f_max: float,
        n_mels: int,
        sample_rate: int,
        dct_type: Optional[str] = 'slaney') -> Tensor:
    if dct_type != "htk" and dct_type != "slaney":
        raise ValueError("DCT type must be either 'htk' or 'slaney'")

    # freq bins
    # Equivalent filterbank construction by Librosa
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    # hertz to mel(f)
    m_min = hz_to_mel(f_min, dct_type)
    m_max = hz_to_mel(f_max, dct_type)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    # mel to hertz(mel)
    f_pts = mel_to_hz(m_pts, dct_type)
    # calculate the difference between each mel point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    # (n_freqs, n_mels + 2)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    if dct_type == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)

    return fb


class MelScaleDelta(nn.Module):
    __constants__ = ['n_mels', 'sample_rate', 'f_min', 'f_max']

    def __init__(self,
                 order,
                 n_mels: int = 128,
                 sample_rate: int = 16000,
                 f_min: float = 0.,
                 f_max: Optional[float] = None,
                 n_stft: Optional[int] = None,
                 dct_type: Optional[str] = 'slaney') -> None:
        super(MelScaleDelta, self).__init__()
        self.order = order
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.dct_type = dct_type

        assert f_min <= self.f_max, 'Require f_min: {} < f_max: {}'.format(
            f_min, self.f_max)

        fb = torch.empty(0) if n_stft is None else create_fb_matrix(
            n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate, self.dct_type)
        self.register_buffer('fb', fb)

    def forward(self, specgram: Tensor) -> Tensor:
        # pack batch
        shape = specgram.size()
        specgram = specgram.reshape(-1, shape[-2], shape[-1])

        if self.fb.numel() == 0:
            tmp_fb = create_fb_matrix(specgram.size(
                1), self.f_min, self.f_max, self.n_mels, self.sample_rate, self.dct_type)
            # Attributes cannot be reassigned outside __init__ so workaround
            self.fb.resize_(tmp_fb.size())
            self.fb.copy_(tmp_fb)

        # (channel, frequency, time).transpose(...) dot (frequency, n_mels)
        # -> (channel, time, n_mels).transpose(...)
        mel_specgram = torch.matmul(
            specgram.transpose(1, 2), self.fb).transpose(1, 2)

        # unpack batch
        mel_specgram = mel_specgram.reshape(
            shape[:-2] + mel_specgram.shape[-2:]).squeeze()

        M = torch.max(torch.abs(mel_specgram))
        if M > 0:
            feat = torch.log1p(mel_specgram / M)
        else:
            feat = mel_specgram

        feat_list = [feat.numpy().T]
        for k in range(1, self.order + 1):
            feat_list.append(savgol_filter(
                feat.numpy(), 9, deriv=k, axis=-1, mode='interp', polyorder=k).T)

        return torch.as_tensor(np.expand_dims(np.stack(feat_list), axis=0))

class Pad(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, wav):
        wav_size = wav.shape[-1]
        pad_size = (self.size - wav_size) // 2
        padded_wav = torch.nn.functional.pad(
            wav, (pad_size, self.size - wav_size - pad_size), mode='constant', value=0)
        return padded_wav

class Rescale(object):
    def __call__(self, input):
        std = torch.std(input, axis=2, keepdims=True,
                        unbiased=False)  # Numpy std is calculated via the Numpy's biased estimator. https://github.com/romainzimmer/s2net/blob/82c38bf80b55d16d12d0243440e34e52d237a2df/data.py#L201
        std.masked_fill_(std == 0, 1)
        return input / std

def collate_fn(data):
    X_batch = torch.cat([d[0] for d in data])
    std = X_batch.std(axis=(0, 2), keepdim=True, unbiased=False)
    X_batch.div_(std)

    y_batch = torch.tensor([d[1] for d in data])

    return X_batch, y_batch

#### Network ####
class LIFWrapper(nn.Module):
    def __init__(self, module, flatten=False):
        super().__init__()
        self.module = module
        self.flatten = flatten

    def forward(self, x_seq: torch.Tensor):
        '''w
        :param x_seq: shape=[batch size, channel, T, n_mel]
        :type x_seq: torch.Tensor
        :return: y_seq, shape=[batch size, channel, T, n_mel]
        :rtype: torch.Tensor
        '''
        # Input: [batch size, channel, T, n_mel]
        y_seq = self.module(x_seq.transpose(0, 2))  # [T, channel, batch size, n_mel]
        if self.flatten:
            y_seq = y_seq.permute(2, 0, 1, 3)  # [batch size, T, channel, n_mel]
            shape = y_seq.shape[:2]
            return y_seq.reshape(shape + (-1,))  # [batch size, T, channel * n_mel]
        else:
            return y_seq.transpose(0, 2)  # [batch size, channel, T, n_mel]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.spike_records = []

        self.train_times = 0
        self.epochs = 0
        self.max_test_acccuracy = 0

        # batch size * delta_order+1 * T * n_mel
        self.conv = nn.Sequential(
            # 101 * 40
            nn.Conv2d(in_channels=delta_order + 1, out_channels=64,
                      kernel_size=(4, 3), stride=1, padding=(2, 1), bias=False),
            LIFWrapper(neuron.LIFNode(tau=10.0 / 7, surrogate_function=surrogate.Sigmoid(alpha=10.), backend=backend,
                                      step_mode='m')),

            # 102 * 40
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(4, 3), stride=1, padding=(6, 3), dilation=(4, 3), bias=False),
            LIFWrapper(neuron.LIFNode(tau=10.0 / 7, surrogate_function=surrogate.Sigmoid(alpha=10.), backend=backend,
                                      step_mode='m')),

            # 102 * 40
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(4, 3), stride=1, padding=(24, 9), dilation=(16, 9), bias=False),
            LIFWrapper(neuron.LIFNode(tau=10.0 / 7, surrogate_function=surrogate.Sigmoid(alpha=10.), backend=backend,
                                      step_mode='m'), flatten=True),
        )

        # [batch size, T, channel * n_mel]
        self.fc = nn.Linear(64 * 40, label_cnt)
        # Register hooks to capture spike sequences
        for name, module in self.named_modules():
            if isinstance(module, LIFWrapper):
                module.register_forward_hook(self.capture_spike_hook)

    def capture_spike_hook(self, module, input, output):

        self.spike_records.append(output)

    def forward(self, x):
        x = self.fc(self.conv(x))  # [batch size, T, #Class]
        return x.mean(dim=1)  # [batch size, #Class]

    def get_spike_records(self):
        return self.spike_records
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-sr', '--sample-rate', type=int, default=16000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-2)
    parser.add_argument('-dir', '--dataset-dir', type=str, default='')
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('-d', '--device', type=str, default='mps')
    args = parser.parse_args()

    sr = args.sample_rate
    n_fft = int(30e-3 * sr)  # 48
    hop_length = int(10e-3 * sr)  # 16
    dataset_dir = args.dataset_dir
    batch_size = args.batch_size
    lr = args.learning_rate
    epoch = args.epoch
    device = args.device

    pad = Pad(size)
    spec = Spectrogram(n_fft=n_fft, hop_length=hop_length)
    melscale = MelScaleDelta(order=delta_order, n_mels=n_mels,
                             sample_rate=sr, f_min=f_min, f_max=f_max, dct_type='slaney')
    rescale = Rescale()

    transform = torchvision.transforms.Compose([pad,
                                                spec,
                                                melscale,
                                                rescale])
    print(label_cnt)

    train_dataset = SPEECHCOMMANDS(
        label_dict, dataset_dir, silence_cnt=2300, url="speech_commands_v0.01", split="train", transform=transform,
        download=True)
    train_sampler = torch.utils.data.WeightedRandomSampler(
        train_dataset.weights, len(train_dataset.weights))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0,
                                  sampler=train_sampler, collate_fn=collate_fn)

    test_dataset = SPEECHCOMMANDS(
        label_dict, dataset_dir, silence_cnt=260, url="speech_commands_v0.01", split="test", transform=transform,
        download=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, collate_fn=collate_fn,
                                 shuffle=False,
                                 drop_last=False)

    net = Net().to(device)
    snn_ckp_dir = os.path.join('/Users/samuelzhou/Desktop/2011/pythonProject10/')

    # Load the pretrain model to execute inferences
    pretrain = torch.load(snn_ckp_dir + 'model.pt', map_location=torch.device('mps'))
    net.load_state_dict(pretrain['model_state_dict'], strict=False)

    optimizer = Adam(net.parameters(), lr=lr)
    gamma = 0.85
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
    warmup_epochs = 1
    print(net)
    writer = SummaryWriter('./logs/')

    criterion = nn.CrossEntropyLoss().to(device)
    test_best_acc = 0

    ##### TEST #####
    net.eval()
    # with torch.no_grad():

def predict(data_np):
    print("data_np.shape")
    print(data_np.shape)
    audio_tensor = torch.FloatTensor(data_np)  # Add batch dimension
    print("audio_tensor.size()")
    print(audio_tensor.size())

    # Apply transformations
    transformed_audio = transform(audio_tensor)
    transformed_audio = transformed_audio.to("mps")
    spike_seq_monitor = monitor.OutputMonitor(net, neuron.LIFNode)
    with torch.no_grad():
        # Make predictions
        out_spikes_counter = net(transformed_audio)
        # with open('spike_records.txt', 'w') as file:
        #     print(net.get_spike_records(), file=file)
        # print(net.get_spike_records())
        # print((net.get_spike_records()[0].shape))
        # print((net.get_spike_records()[1].shape))
        # print((net.get_spike_records()[2].shape))

        probabilities_percent = torch.softmax(out_spikes_counter, dim=1) * 100
        probabilities_percent_np = probabilities_percent.cpu().numpy()
        probabilities_percent_rounded = np.round(probabilities_percent_np, 2)
        # print(probabilities_percent_rounded.flatten())
        #print(net.get_spike_records())

        #print(f'spike_seq_monitor.records=\n{spike_seq_monitor.records}')

        reset_net(net)
    return probabilities_percent_rounded.flatten()

def preprocess_for_visualization(tensor):
    # Adapt this based on the specifics of each tensor.
    if tensor.dim() == 4:  # For tensors with spatial dimension
        processed_tensor = tensor.sum(dim=-1).squeeze()  # Sum over spatial dimension, squeeze batch dimension
    else:  # For the flattened tensor
        # Example reshape back to [time, channels] format, adjust based on actual channel/spatial relationship
        processed_tensor = tensor.squeeze()  # Squeeze batch dimension
    return processed_tensor

class AudioProcessingThread(QThread):
    # Signal to update the plot
    update_waveform_signal = pyqtSignal(np.ndarray)
    update_spectrogram_signal = pyqtSignal(np.ndarray)
    update_prediction_signal = pyqtSignal(object)
    def __init__(self, parent=None):
        super(AudioProcessingThread, self).__init__(parent)
        self.CHUNK = 2000
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  # Use 16000 to match the SNN model's expectation
        self.RECORD_SECONDS = 1  # Record for 1 second
        self.BUNCH = self.RATE * self.RECORD_SECONDS / self.CHUNK
        self.running = False
        self.N_MELS = 128
        self.n_fft = int(30e-3 * self.RATE)  # Window size for STFT
        self.hop_length = int(10e-3 * self.RATE)  # Hop length for STFT
        self.n_mels = 40
        self.f_max = 4000
        self.f_min = 20
        self.delta_order = 0
        self.window=20
        self.predictions_deque = {i: deque(maxlen=self.window) for i in range(12)}

    def handle_prediction(self, data_np):
        # This method runs in a separate thread
        new_predictions = predict(data_np)
        print("update_prediction right after net outcome", time.time() - start_time)
        for i in range(12):
            self.predictions_deque[i].append(new_predictions[i])
            # print(self.predictions_deque[i])
        self.update_prediction_signal.emit(self.predictions_deque)


    def run(self):
        self.running = True

        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT, channels=1, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)

        while self.running:
            frames = []
            global start_time
            start_time = time.time()
            for _ in range(0,int(self.BUNCH)):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)
                data_np1 = np.frombuffer(data, dtype=np.int16)
                # if _%2==0:
                #     self.update_waveform_signal.emit(data_np1)
                #     time.sleep(0.02)

                self.update_waveform_signal.emit(data_np1)
                time.sleep(0.125)


            print(len(frames))
            data_np = np.frombuffer(b''.join(frames), dtype=np.int16).copy()
            print(data_np.shape)
            print("update_prediction before function", time.time()-start_time)

            # new_predictions = predict(data_np)
            # for i in range(12):
            #     self.predictions_deque[i].append(new_predictions[i])
            #     # print(self.predictions_deque[i])
            # self.update_prediction_signal.emit(self.predictions_deque)
            prediction_thread = Thread(target=self.handle_prediction, args=(data_np,))
            prediction_thread.start()
            print("update_prediction after function", time.time()-start_time)

            data_float = data_np/ 32768.0
            # Generate Mel spectrogram

            S = librosa.feature.melspectrogram(y=data_float, sr=self.RATE, n_mels=self.N_MELS, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            print("update_spec before function", time.time()-start_time)
            self.update_spectrogram_signal.emit(S_dB)
            print("update_spec after function", time.time()-start_time)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def stop(self):
        self.running = False

class AudioPredictionGUI(QMainWindow):

    def __init__(self, parent=None):
        super(AudioPredictionGUI, self).__init__(parent)
        self.setWindowTitle('Real-Time Audio Prediction')

        # Central widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.CHUNK = 2000
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.window = 20  # Define the window size
        self.RATE = 16000  # Use 16000 to match the SNN model's expectation
        self.RECORD_SECONDS = 1  # Record for 1 second
        self.BUNCH = self.RATE* self.RECORD_SECONDS/ self.CHUNK
        # Constants for processing
        self.n_fft = int(30e-3 * self.RATE)  # Window size for STFT
        self.hop_length = int(10e-3 * self.RATE)  # Hop length for STFT
        self.n_mels = 40
        self.f_max = 4000
        self.f_min = 20
        self.delta_order = 0
        self.size = self.RATE * self.RECORD_SECONDS

        # Matplotlib Figure
        self.figure = Figure(figsize=(10, 9), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.ax1 = self.figure.add_subplot(231)
        self.ax2 = self.figure.add_subplot(232)
        self.ax3 = self.figure.add_subplot(233)
        self.ax4 = self.figure.add_subplot(234)
        self.ax5 = self.figure.add_subplot(235)
        self.ax6 = self.figure.add_subplot(236)

        self.x_audio = np.arange(0, 2 * self.CHUNK, 2)
        self.line_audio, = self.ax1.plot(self.x_audio, np.random.rand(self.CHUNK), '-', lw=2)
        self.ax1.set_title('AUDIO WAVEFORM')
        self.ax1.set_xlim(0, 2 * self.CHUNK)
        self.ax1.set_ylim(-5000, 5000)

        # Set up the plot
        self.predictions_deque = {i: deque(maxlen=self.window) for i in range(12)}
        self.lines = [self.ax6.plot([], [])[0] for _ in range(12)]
        labels = ["Yes", "Stop", "No", "Right", "Up", "Left", "On", "Down", "Off", "Go", "Other", "Silence"]
        for line, label in zip(self.lines, labels):
            line.set_label(label)

        self.S = None
        self.colorbar = None

        self.ax6.set_title('OUTPUT PREDICTION')
        self.ax6.set_xlabel('Time step')
        self.ax6.set_ylabel('Percent')
        self.ax6.set_xlim(0, self.window)
        self.ax6.set_ylim(0, 105)

        self.ax6.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

        self.ax2.set_title('MEL SPECTROGRAM')

        # Buttons
        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.start_prediction)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop_prediction)
        self.layout.addWidget(self.stop_button)

        # Thread for background processing
        self.updateTimer = QTimer(self)
        self.updateTimer.timeout.connect(self.updateGUI)
        self.updateTimer.start(100)  # Adjust
        self.pendingUpdate = False
        self.audio_thread = AudioProcessingThread()
        self.audio_thread.update_waveform_signal.connect(self.update_audio_waveform)
        self.audio_thread.update_spectrogram_signal.connect(self.update_spectrogram)
        self.audio_thread.update_prediction_signal.connect(self.update_prediction)

    def updateGUI(self):
        if self.pendingUpdate:
            # Update the GUI with the latest data
            self.canvas.draw_idle()
            print("GUI Updated")
            self.pendingUpdate = False

    def markForUpdate(self):
        self.pendingUpdate = True
    def start_prediction(self):

        if not self.audio_thread.isRunning():
            self.audio_thread.start()

    def stop_prediction(self):
        self.audio_thread.stop()
        print("Thread stopped")

    def update_audio_waveform(self, data_np):

        # Update plot with new audio data
        self.line_audio.set_ydata(data_np)
        self.markForUpdate()


        print("update_audio_waveform",time.time()-start_time)
    def update_spectrogram(self, S_dB):

        spectrogram_img = librosa.display.specshow(S_dB, sr=self.RATE, x_axis='time', y_axis='mel', fmax=8000,
                                                   ax=self.ax2)
        # self.canvas.draw_idle()
        self.markForUpdate()

        print("update_spectrogram", time.time()-start_time)
    def update_prediction(self,predictions_deque):
        # self.canvas.draw()

        # Append new predictions to the right of the deque, old predictions will be removed from the left
        spike_records=net.get_spike_records()
        #print(len(spike_records))
        for idx, tensor in enumerate(spike_records):
            # Preprocess the tensor for visualization
            #print(tensor.shape)
            processed_tensor = preprocess_for_visualization(tensor)
            # Selecting the appropriate axis based on idx
            if idx == 0:
                ax = self.ax3
            elif idx == 1:
                ax = self.ax4
            elif idx == 2:
                ax = self.ax5
            else:
                # Optionally handle cases where idx > 2, or simply continue to ignore
                continue  # Skip further processing for indices beyond 2

            # Common visualization code, using 'ax' determined above
            ax.cla()  # Clear the axis to ensure previous images/annotations are removed
            ax.imshow(processed_tensor.cpu().numpy(), aspect="auto", cmap="viridis", interpolation='nearest')
            ax.set_title(f'Layer {idx + 1} Spike Activity')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Channels/Neurons')



        net.spike_records = []
        self.markForUpdate()

        # for i in range(12):
        #     self.predictions_deque[i].append(new_predictions[i])
        #     # print(self.predictions_deque[i])
        for i, line in enumerate(self.lines):
            line.set_data(np.arange(len(predictions_deque[i])), list(predictions_deque[i]))
        # self.canvas.draw_idle()
        print("update_prediction in function", time.time()-start_time)
        self.markForUpdate()


        # self.ax6.relim()  # Recalculate limits
        # self.ax6.autoscale_view(True,True,True)  # Rescale the view based on the new line data
        # self.canvas.draw_idle()  # Update the canvas with the new line data
    def closeEvent(self, event):
        self.audio_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = AudioPredictionGUI()
    mainWin.show()
    sys.exit(app.exec_())

