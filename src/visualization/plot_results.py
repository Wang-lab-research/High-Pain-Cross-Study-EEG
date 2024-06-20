from glob import glob
import pickle
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from mne.time_frequency import tfr_array_morlet, AverageTFRArray
import seaborn as sns
from typing import Dict, List, Union
from IPython.display import clear_output

sns.set(style="white", font_scale=1.5)


class Visualizer:
    def __init__(
        self,
        sfreq: int,
        roi_names: List[str],
        roi_acronyms: List[str],
        freq_bands: Dict[str, List[int]],
        output_path: str,
        model_name: str,
    ):
        self.sfreq = sfreq
        self.roi_names = roi_names
        self.roi_acronyms = roi_acronyms
        self.freq_bands = freq_bands
        self.output_path = output_path
        self.model_name = model_name
        self.yes_list = []
        self.no_list = []
        self.maybe_list = []


    def _compute_tfr(self, subjects: Union[Subject, SubjectGroup]) -> AverageTFRArray:
        epochs, _, _, stc_epo_array, stc_resting = self._load_complete_data(subjects)

        freqs = np.logspace(*np.log10([1, 100]), num=50)
        n_cycles = freqs / 2.0

        info = mne.create_info(
            ch_names=self.roi_acronyms, sfreq=self.sfreq, ch_types="eeg"
        )

        power = tfr_array_morlet(
            stc_epo_array,
            sfreq=self.sfreq,
            freqs=freqs,
            n_cycles=n_cycles,
            output="avg_power",
            zero_mean=False,
        )

        tfr = AverageTFRArray(
            info=info,
            data=power,
            times=epochs.times,
            freqs=freqs,
            nave=stc_epo_array.shape[0],
        )

        return tfr

    def _plot_tfr(
        self,
        tfr: AverageTFRArray,
        baseline: tuple,
        title: str,
        time_range: tuple = (-0.2, 0.8),
        vlim: tuple = (None, None),
        orientation: str = "horizontal",
    ):
        if orientation == "vertical":
            fig, axes = plt.subplots(6, 2, figsize=(12, 16), sharex=False, sharey=True)
        elif orientation == "horizontal":
            fig, axes = plt.subplots(2, 6, figsize=(22, 6), sharex=False, sharey=False)
            
        for i, roi in enumerate(self.roi_acronyms):
            col = i // 6 if orientation == "vertical" else i % 6
            row = i % 6 if orientation == "vertical" else i // 6
            ax = axes[row, col]
            tfr.plot(
                baseline=baseline,
                tmin=time_range[0],
                tmax=time_range[1],
                picks=roi,
                mode="zscore",
                title=title,
                yscale="linear",
                colorbar=False,
                vlim=vlim,
                cmap="turbo",
                axes=ax,
                show=False,
            )
            ax.set_title(roi)
        
            im = ax.images[-1]  # The last image plotted on the axes
    
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_ticks([-3, 3])
            if col > 0:
                ax.set_ylabel("")
            if row == 0:
                ax.set_xlabel("")
            if i == 0 and orientation == "vertical":
                ax.set_yticks([0,20,40,60,80,100])
                ax.set_yticklabels([str(x) for x in [0,20,40,60,80,100]])
            elif orientation == "horizontal":
                ax.set_yticks([0,20,40,60,80,100])
                ax.set_yticklabels([str(x) for x in [0,20,40,60,80,100]])

            # add stimulus onset line
            ax.axvline(x=0, color="red", linestyle="--", label="Stimulus Onset")

        fig.tight_layout()
        plt.show()
        return fig

    def _plot_trace(
        self,
        subjects: Union[Subject, SubjectGroup],
        epochs,
        evoked_data_arrays,
        sem_epochs_per_sub,
        channel: str,
        time_range: tuple,
    ):
        evoked_averaged = np.nanmean(evoked_data_arrays, axis=0)
        sem_averaged = np.nanmean(sem_epochs_per_sub, axis=0)

        sfreq = self.sfreq
        total_duration = evoked_averaged.shape[1] / sfreq
        time_min = -total_duration / 2

        sample_start = int((time_range[0] - time_min) * sfreq)
        sample_end = int((time_range[1] - time_min) * sfreq)

        sample_start = max(sample_start, 0)
        sample_end = min(sample_end, evoked_averaged.shape[1])

        timepoints = np.linspace(
            time_range[0], time_range[1], sample_end - sample_start
        )

        # Find the index of the channel
        channel_index = (
            epochs.info["ch_names"].index(channel)
            if channel in epochs.info["ch_names"]
            else epochs.info["ch_names"].index(f"{channel.upper()}")
        )

        # Get data within the time range
        evoked_averaged = evoked_averaged[channel_index, sample_start:sample_end] * 1e7
        sem_averaged = sem_averaged[channel_index, sample_start:sample_end] * 1e7

        fig = plt.figure(figsize=(10, 5))
        plt.plot(
            timepoints,
            evoked_averaged,
            label=f"Channel {channel}",
        )

        # Plot SEM as shaded area
        plt.fill_between(
            timepoints,
            evoked_averaged - sem_averaged,
            evoked_averaged + sem_averaged,
            color="b",
            alpha=0.4,
            label="SEM",
        )

        plt.axvline(x=0, color="red", linestyle="--", label="Stimulus Onset")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (µV)")
        plt.title(
            f"Grand Average ERP ({subjects.group})"
            if isinstance(subjects, Subject)
            else f"Grand Average ERP (Group-Averaged {subjects.subjects[0].group})"
        )
        plt.tight_layout()
        plt.legend()
        plt.xlim(time_range)
        plt.show()
        return fig

    def plot_TFR_and_trace(
        self,
        subjects: Union[Subject, SubjectGroup],
        channel="Fp1",
        baseline=(-2.5, 0.0),
        time_range=(-0.2, 0.8),
        vlim=None,
        orientation="vertical",
    ):
        tfr = self._compute_tfr(subjects)

        epochs, evoked_data_arrays, sem_epochs_per_sub, stc_epo_array, stc_resting = (
            self._load_complete_data(subjects)
        )

        title = (
            f"Time-Frequency Representation of Evoked  Response ({subjects.group})"
            if isinstance(subjects, Subject)
            else f"Time-Frequency Representation of Group-Averaged Evoked Response ({subjects.subjects[0].group})"
        )

        tfr_fig = self._plot_tfr(tfr, baseline, title, time_range, vlim, orientation)
        save_fig_path = os.path.join(self.stc_path, f"{baseline}")
        os.makedirs(save_fig_path, exist_ok=True)
        # if isinstance(subjects, Subject):
        #     tfr_fig.savefig(os.path.join(save_fig_path, f"{subjects.subject_id}_epochs_tfr.png"),
        #      dpi=300)))
        if isinstance(subjects, SubjectGroup):
            tfr_fig.savefig(os.path.join(save_fig_path, f"{subjects.group}_epochs_tfr.png"),
                                         dpi=300)
            
        trace_fig = self._plot_trace(
            subjects,
            epochs,
            evoked_data_arrays,
            sem_epochs_per_sub,
            channel,
            time_range,
        )
        # if isinstance(subjects, Subject):
        #     trace_fig.savefig(os.path.join(save_fig_path, f"{subjects.subject_id}_epochs_trace.png"),
        #      dpi=300))
        if isinstance(subjects, SubjectGroup):
            trace_fig.savefig(os.path.join(save_fig_path, f"{subjects.group}_epochs_trace.png"),
                              dpi=300)
        
        return tfr, evoked_data_arrays
