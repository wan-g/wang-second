# This is the processing script of DREAMER dataset

import _pickle as cPickle
import pandas as pd
from train_model import *
import scipy.io as sio
from scipy.signal import decimate
import torchvision.transforms as transforms

import torch
from utils import generate_TS_channel_order
import os

class PrepareData:
    def __init__(self, args):
        # init all the parameters here
        # arg contains parameter settings
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        self.data_path = args.data_path
        self.label_type = args.label_type
        self.original_order = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']  # 14 , 'M1', 'M2'

        self.TS_order = generate_TS_channel_order(self.original_order)   # generate proper channel orders for the asymmetric spatial layer in TSception

        self.graph_type = args.graph_type

    def run(self, sub_clip_list, split=False, expand=True, feature=False):
        """
        Parameters
        ----------
        sub_clip_list: the subjects and clips ID needed to be processed
        split: (bool) whether to split one trial's data into shorter segment
        expand: (bool) whether to add an empty dimension for CNN
        feature: (bool) whether to extract features or not

        Returns
        -------
        The processed data will be saved './data_<data_format>_<dataset>_<label_type>/sub0.hdf'
        """
        for sub in sub_clip_list:
                data_, label_ = self.load_data_per_subject(sub)
                # select label type here
                label_ = self.label_selection(label_)

                if split:
                    data_, label_ = self.split(
                    data=data_, label=label_, segment_length=self.args.segment,
                    overlap=self.args.overlap, sampling_rate=self.args.sampling_rate)

                if expand:
                    # expand one dimension for deep learning(CNNs)
                    data_ = np.expand_dims(data_, axis=-3)

                print('Data and label prepared for sub{}!'.format(sub))
                print('data:' + str(data_.shape) + ' label:' + str(label_.shape))
                print(f"The label: {label_}")
                print('----------------------')
                self.save(data_, label_, sub)

    def load_data_per_subject(self, sub):
        """
        This function loads the target subject's original file
        Parameters
        ----------
        sub: which subject to load
        clip: which clip to load

        Returns
        -------
        data: (trial, 14, time) label: (trial, 3)
        """

        raw_path = os.path.join(self.data_path, 'DREAMER')
        raw = sio.loadmat(raw_path, verify_compressed_data_integrity=False)
        trial_len = len(raw['DREAMER'][0, 0]['Data'][0, 0]['EEG'][0, 0]['stimuli'][0, 0])  #18
        # subject_len = len(raw['DREAMER'][0, 0]['Data'][0])  # 23

        data_EEG = []
        data = []
        label = []

        # loop for each trial
        for trial_id in range(trial_len):
            # extract baseline signals
            trial_baseline_sample = raw['DREAMER'][0, 0]['Data'][0, sub]['EEG'][0, 0]['baseline'][0, 0][trial_id, 0]
            trial_baseline_sample = trial_baseline_sample[:, :14].swapaxes(1, 0)  # channel(14), timestep(61*128)
            trial_baseline_sample = trial_baseline_sample[:, : 61 * 128].reshape(14, 61, 128).mean(axis=1)  # channel(14), timestep(128)


            # record the common meta info
            # trial_meta_info = {'subject_id': sub, 'trial_id': trial_id}  # dictionary
            # trial_meta_info['valence'] = raw['DREAMER'][0, 0]['Data'][0, sub]['ScoreValence'][0, 0][trial_id, 0]
            # trial_meta_info['arousal'] = raw['DREAMER'][0, 0]['Data'][0, sub]['ScoreArousal'][0, 0][trial_id, 0]
            # trial_meta_info['dominance'] = raw['DREAMER'][0, 0]['Data'][0, sub]['ScoreDominance'][0, 0][trial_id, 0]
            valence = raw['DREAMER'][0, 0]['Data'][0, sub]['ScoreValence'][0, 0][trial_id, 0]
            arousal = raw['DREAMER'][0, 0]['Data'][0, sub]['ScoreArousal'][0, 0][trial_id, 0]

            data_stimuli_ecg = raw['DREAMER'][0, 0]['Data'][0, sub]["ECG"][0, 0]["stimuli"][0, 0][trial_id, 0]
            data_stimuli_ecg = decimate(data_stimuli_ecg, 4, axis=0).swapaxes(1, 0)  # 512Hz降采样到128Hz

            trial_samples = raw['DREAMER'][0, 0]['Data'][0, sub]['EEG'][0, 0]['stimuli'][0, 0][trial_id, 0]  # [T, C]
            trial_samples = trial_samples[:, :14].swapaxes(1, 0)  # channel(14), timestep(n*128)

            if data_stimuli_ecg.shape[1] < 7680:
                data_ECG = np.pad(data_stimuli_ecg, ((0, 0), (0, 7680 - data_stimuli_ecg.shape[1])), mode='constant',
                                     constant_values=data_stimuli_ecg[:, -1][:, np.newaxis])
            else:
                data_ECG = data_stimuli_ecg


            if trial_samples.shape[1] >= 7680:
                combined_data = np.concatenate((trial_samples[:, -7680:], data_ECG[:, -7680:]), axis=0)
                # data_EEG.append(trial_samples[:, -7680:])
                data.append(combined_data)
                # data_EEG.append(trial_samples[:, 0:7680])

            trial_label = [valence, arousal]
            trial_label = np.array(trial_label)

            label.append(trial_label)

        # Pad arrays with zeros or NaNs to match the longest length
        # max_len = max(arr.shape[1] for arr in data_EEG)
        # padded_data_EEG = [np.pad(arr, ((0, 0), (0, max_len - arr.shape[1]))) for arr in data_EEG]

        combined_data = np.stack(data, axis=0)  # [trial, c, t] 堆叠函数
        data = self.reorder_channel(data=combined_data, graph=self.graph_type)
        label = np.stack(label, axis=0)

        print('data:' + str(data.shape) + ' label:' + str(label.shape))

        return data, label

    def reorder_channel(self, data, graph):
        """
        This function reorder the channel according to different graph designs
        Parameters
        ----------
        data: (trial, channel, data)
        graph: graph type

        Returns
        -------
        reordered data: (trial, channel, data)
        """
        if graph == 'TS':
            graph_idx = self.TS
        elif graph == 'O':
            graph_idx = self.original_order

        idx = []

        for chan in graph_idx:
            idx.append(self.original_order.index(chan))

        return data[:, idx, :]


    def label_selection(self, label):
        # V A D
        """
        This function: 1. selects which dimension of labels to use
                       2. create binary label
        Parameters
        ----------
        label: (trial, 4)

        Returns
        -------
        label: (trial,)
        """
        if self.label_type == 'V':
            label = label[:, 0]
        elif self.label_type == 'A':
            label = label[:, 1]
        elif self.label_type == 'D':
            label = label[:, 2]
            
        if self.args.num_class == 2:
            label = np.where(label <= 3.0, 0, label)
            label = np.where(label > 3.0, 1, label)
            print('Binary label generated!')
        return label

    def save(self, data, label, sub):
        """
        This function save the processed data into target folder
        Parameters
        ----------
        data: the processed data
        label: the corresponding label
        sub: the subject ID

        Returns
        -------
        None
        """
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        save_path = osp.join(save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        name = 'sub' + str(sub) + '.hdf'
        save_path = osp.join(save_path, name)
        dataset = h5py.File(save_path, 'w')
        dataset['data'] = data
        dataset['label'] = label
        dataset.close()

    def split(self, data, label, segment_length=1, overlap=0, sampling_rate=256):
        """
        This function split one trial's data into shorter segments
        Parameters
        ----------
        data: (trial, channel, data)
        label: (trial,)
        segment_length: how long each segment is (e.g. 1s, 2s,...)
        overlap: overlap rate
        sampling_rate: sampling rate

        Returns
        -------
        data:(tiral, num_segment, channel, segment_legnth)
        label:(trial, num_segment,)
        """
        data_shape = data.shape
        step = int(segment_length * sampling_rate * (1 - overlap))  # 4*128=512
        data_segment = sampling_rate * segment_length  # 512
        data_split = []

        number_segment = int((data_shape[2] - data_segment) // step)  #
        for i in range(number_segment + 1):
            data_split.append(data[:, :, (i * step):(i * step + data_segment)])
        data_split_array = np.stack(data_split, axis=1)
        label = np.stack([np.repeat(label[i], int(number_segment + 1)) for i in range(len(label))], axis=0)
        print("The data and label are split: Data shape:" + str(data_split_array.shape) + " Label:" + str(
            label.shape))
        data = data_split_array
        assert len(data) == len(label)
        return data, label
