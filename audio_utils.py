# audio_utils.py
# Robust audio loading, extension, and Doppler application utilities for Doppler simulator.
# Exports: load_original_audio, apply_doppler_to_audio_fixed,
#          apply_doppler_to_audio_fixed_alternative, apply_doppler_to_audio_fixed_advanced,
#          normalize_amplitudes, save_audio, SR

import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from scipy.interpolate import interp1d
import os

SR = 22050  # sample rate used throughout
DEFAULT_EXTEND_OVERLAP_SEC = 0.1  # default overlap when extending audio

# Map vehicle type to fallback filename (if you store assets elsewhere, update these paths)
AUDIO_FILE_MAP = {
    'car': 'static/horn.mp3',
    'train': 'static/train.mp3',
    'flight': 'static/flight.mp3',
    'drone': 'static/drone.mp3'
}


# ---------------------------
# Loading / extending helpers
# ---------------------------
def load_original_audio(audio_type='car', duration=5.0):
    """
    Load the original audio for a given vehicle type and ensure it's exactly `duration` seconds.
    - audio_type: key in AUDIO_FILE_MAP (e.g., 'car', 'drone')
    - duration: desired seconds (float)
    Returns: mono numpy array of length SR * duration (dtype float32)
    """
    target_samples = int(SR * float(duration))
    filename = AUDIO_FILE_MAP.get(audio_type, AUDIO_FILE_MAP['car'])

    # Try to load using librosa; if fails, generate a synthetic fallback
    try:
        y, original_sr = librosa.load(filename, sr=SR, mono=True)
        # Trim or extend as needed
        if len(y) >= target_samples:
            return y[:target_samples].astype(np.float32)
        else:
            return extend_audio_with_overlap(y, duration, SR)
    except Exception as e:
        # Log and generate fallback synthetic audio
        print(f"Warning: failed to load '{filename}' ({e}). Generating fallback '{audio_type}' sound.")
        return _generate_fallback_sound(audio_type, duration)


def _generate_fallback_sound(audio_type, duration):
    """
    Generate a simple synthetic audio signal when asset not available.
    Keeps character distinct for car/train/drone.
    """
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    if audio_type == 'car':
        audio = (np.sin(2*np.pi*440*t) +
                 0.7*np.sin(2*np.pi*880*t) +
                 0.4*np.sin(2*np.pi*1320*t))
        envelope = np.exp(-t * 0.3) * 0.5 + 0.5
    elif audio_type == 'train':
        audio = (np.sin(2*np.pi*220*t) +
                 0.8*np.sin(2*np.pi*110*t) +
                 0.6*np.sin(2*np.pi*330*t))
        rhythm = 1.0 + 0.3*np.sin(2*np.pi*8*t)
        audio = audio * rhythm
        envelope = 0.8 + 0.2*np.sin(2*np.pi*2*t)
    elif audio_type == 'flight':
        audio = (0.6*np.sin(2*np.pi*800*t) +
                 0.8*np.sin(2*np.pi*1200*t) +
                 0.4*np.sin(2*np.pi*1600*t))
        turbine = 1.0 + 0.2*np.sin(2*np.pi*15*t)
        audio = audio * turbine
        envelope = 0.9 + 0.1*np.sin(2*np.pi*3*t)
    else:  # drone
        # Drone-like rotor: rich high-frequency harmonics + mild amplitude modulation
        base = (0.8*np.sin(2*np.pi*400*t) +
                0.6*np.sin(2*np.pi*800*t) +
                0.4*np.sin(2*np.pi*1200*t) +
                0.2*np.sin(2*np.pi*1600*t))
        tremolo = 1.0 + 0.12*np.sin(2*np.pi*8*t)  # rotor blade modulation
        audio = base * tremolo
        envelope = 0.95 - 0.05*np.exp(-t*1.0)

    out = audio * envelope
    # Normalize moderate amplitude
    out = out.astype(np.float32)
    maxv = np.max(np.abs(out))
    if maxv > 0:
        out = out / maxv * 0.8
    return out


def extend_audio_with_overlap(original_audio, target_duration, sample_rate):
    """
    Extend a shorter clip to reach target_duration (in seconds) by repeating it with small
    crossfades to create a seamless loop.
    """
    original = np.asarray(original_audio, dtype=np.float32)
    original_len = len(original)
    target_len = int(sample_rate * float(target_duration))

    if original_len == 0:
        # Fallback to synthetic short pulse
        t = np.linspace(0, target_duration, target_len, endpoint=False)
        return 0.5 * np.sin(2*np.pi*440*t).astype(np.float32)

    if original_len >= target_len:
        return original[:target_len]

    overlap_sec = DEFAULT_EXTEND_OVERLAP_SEC
    overlap_samples = int(sample_rate * overlap_sec)
    overlap_samples = max(1, min(overlap_samples, original_len // 4))

    out = np.zeros(target_len, dtype=np.float32)
    out[:original_len] = original
    write_pos = original_len

    iter_count = 1
    while write_pos < target_len:
        remaining = target_len - write_pos
        # compute region to copy from original
        copy_len = min(original_len - overlap_samples, remaining)
        start_write = max(0, write_pos - overlap_samples)
        end_write = start_write + overlap_samples + copy_len

        # crossfade overlap region
        if overlap_samples > 0:
            fade_out = np.linspace(1.0, 0.0, overlap_samples, endpoint=False).astype(np.float32)
            fade_in = np.linspace(0.0, 1.0, overlap_samples, endpoint=False).astype(np.float32)
            # fade existing region
            out[start_write:start_write+overlap_samples] *= fade_out
            # add faded-in content from original start
            out[start_write:start_write+overlap_samples] += original[:overlap_samples] * fade_in
        # add non-overlapping portion
        if copy_len > 0:
            src_start = overlap_samples
            src_end = overlap_samples + copy_len
            dest_start = start_write + overlap_samples
            dest_end = dest_start + copy_len
            out[dest_start:dest_end] = original[src_start:src_end]

        write_pos = end_write
        iter_count += 1

    # final gentle fade-out to avoid clicks
    fade_len = min(int(sample_rate * 0.05), target_len // 50)
    if fade_len > 0:
        fade = np.linspace(1.0, 0.0, fade_len, endpoint=False).astype(np.float32)
        out[-fade_len:] *= fade

    # normalize moderate amplitude
    maxv = np.max(np.abs(out))
    if maxv > 0:
        out = out / maxv * 0.8

    return out.astype(np.float32)


# ---------------------------
# Doppler application methods
# ---------------------------
def apply_true_doppler_shift(original_audio, freq_ratios, amplitudes):
    """
    Time-domain sample-rate warping approach. Interpolates an instantaneous sample index
    trajectory derived from freq_ratios and resamples the original audio along that trajectory.
    Produces clear sweeps in spectrograms.
    """
    original = np.asarray(original_audio, dtype=np.float32)
    N = len(original)
    # desired output length equals original length (caller can trim/pad)
    out_len = N

    # create per-sample time mapping
    # freq_ratios is per-frame values; map them to per-sample multiplier curve
    curve_x = np.linspace(0, out_len - 1, num=len(freq_ratios))
    interp_freq = interp1d(curve_x, freq_ratios, kind='linear', bounds_error=False, fill_value=(freq_ratios[0], freq_ratios[-1]))
    interp_amp = interp1d(curve_x, amplitudes, kind='linear', bounds_error=False, fill_value=(amplitudes[0], amplitudes[-1]))

    freq_curve = interp_freq(np.arange(out_len))
    amp_curve = interp_amp(np.arange(out_len))

    # smooth small jitter but don't over-smooth
    if len(freq_curve) > 50:
        try:
            window = 11 if len(freq_curve) >= 11 else (len(freq_curve)//2*2+1)
            freq_curve = signal.savgol_filter(freq_curve, window, 1)
            amp_curve = signal.savgol_filter(amp_curve, window, 1)
        except Exception:
            pass

    # cumulative sample progression (sum of instantaneous speed multipliers)
    # cumulative_samples[n] gives the position (in original sample indices) to sample at output index n
    cumulative = np.cumsum(freq_curve)
    # normalize to fit within original audio index range
    if cumulative[-1] <= 0:
        cumulative[-1] = 1.0
    cumulative = cumulative * ((N - 1) / cumulative[-1])

    # Resample original at fractional indices using np.interp
    orig_indices = np.arange(N)
    output = np.interp(cumulative, orig_indices, original).astype(np.float32)

    # Apply amplitude envelope
    output *= amp_curve.astype(np.float32)

    # Moderate normalization to avoid clipping
    maxv = np.max(np.abs(output)) if output.size else 1.0
    if maxv > 0:
        output = output / maxv * 0.98

    return output


def apply_spectral_doppler_shift(original_audio, freq_ratios, amplitudes):
    """
    STFT-based spectral warping. This is heavier but can produce good visual sweeps.
    We use moderate STFT parameters to emphasize temporal resolution.
    """
    original = np.asarray(original_audio, dtype=np.float32)
    N = len(original)

    # Map per-frame freq_ratios/amplitudes to STFT frames
    n_fft = 1024
    hop_length = 256
    stft = librosa.stft(original, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft)
    phase = np.angle(stft)

    n_frames = stft.shape[1]
    frame_positions = np.linspace(0, N - 1, n_frames)
    freq_curve_interp = interp1d(np.linspace(0, N - 1, len(freq_ratios)), freq_ratios, kind='linear', bounds_error=False, fill_value=(freq_ratios[0], freq_ratios[-1]))
    amp_curve_interp = interp1d(np.linspace(0, N - 1, len(amplitudes)), amplitudes, kind='linear', bounds_error=False, fill_value=(amplitudes[0], amplitudes[-1]))

    freq_curve_frames = freq_curve_interp(frame_positions)
    amp_curve_frames = amp_curve_interp(frame_positions)

    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    output_stft = np.zeros_like(stft, dtype=np.complex64)

    # For each frame, remap energy from old bins to new bins based on ratio
    for fi in range(n_frames):
        ratio = float(freq_curve_frames[fi])
        amp_fac = float(amp_curve_frames[fi])

        mag_frame = mag[:, fi]
        ph_frame = phase[:, fi]

        # New magnitude array init
        new_mag = np.zeros_like(mag_frame)
        new_ph = np.zeros_like(ph_frame)

        # For each source bin, compute where it should land after scaling
        # Use linear interpolation of magnitude into the target bins
        target_bin_positions = freqs * ratio
        # convert target freqs to fractional bin indices in current freqs array
        target_idx = np.interp(target_bin_positions, freqs, np.arange(len(freqs)))
        # distribute each source bin's magnitude into lower/upper bins
        lower = np.floor(target_idx).astype(int)
        upper = np.ceil(target_idx).astype(int)
        weights = target_idx - lower

        # clip indices
        L = len(new_mag)
        for src_bin in range(1, len(freqs)):  # skip DC at index 0 for stability
            tgt_l = lower[src_bin]
            tgt_u = upper[src_bin]
            w = weights[src_bin]
            val = mag_frame[src_bin]
            if 0 <= tgt_l < L:
                new_mag[tgt_l] += val * (1.0 - w)
                new_ph[tgt_l] = ph_frame[src_bin]
            if 0 <= tgt_u < L:
                new_mag[tgt_u] += val * w
                new_ph[tgt_u] = ph_frame[src_bin]

        # Store complex frame with amplitude modulation
        output_stft[:, fi] = (new_mag * amp_fac) * np.exp(1j * new_ph)

    # ISTFT back to time domain
    out = librosa.istft(output_stft, hop_length=hop_length, length=N).astype(np.float32)

    # Normalize
    maxv = np.max(np.abs(out)) if out.size else 1.0
    if maxv > 0:
        out = out / maxv * 0.98

    return out


def apply_phase_modulation_doppler(original_audio, freq_ratios, amplitudes):
    """
    Phase modulation approach: constructs an analytic signal, modulates its phase trajectory
    according to freq_ratios, and returns the real part. This yields clear instantaneous-frequency sweeps.
    """
    original = np.asarray(original_audio, dtype=np.float32)
    N = len(original)

    # Interpolate freq_ratios/amplitudes to sample resolution
    x_src = np.linspace(0, N - 1, num=len(freq_ratios))
    freq_interp = interp1d(x_src, freq_ratios, kind='linear', bounds_error=False, fill_value=(freq_ratios[0], freq_ratios[-1]))
    amp_interp = interp1d(x_src, amplitudes, kind='linear', bounds_error=False, fill_value=(amplitudes[0], amplitudes[-1]))

    freq_curve = freq_interp(np.arange(N))
    amp_curve = amp_interp(np.arange(N))

    # Convert ratio curve to deviation around 1.0 and compute phase increments
    # instantaneous frequency multiplier is freq_curve; cumulative phase is integral
    dt = 1.0 / SR
    inst_freq = freq_curve  # multiplier of nominal sampling progression
    # cumulative sample index progression
    cumulative = np.cumsum(inst_freq)
    # normalize to original length
    cumulative = cumulative * ((N - 1) / cumulative[-1])

    # Build analytic signal and resample phase-modulated version
    analytic = signal.hilbert(original)
    # phase trajectory for each output sample: 2*pi * cumulative_index_mapping / SR
    phase_traj = 2.0 * np.pi * cumulative / SR
    modulation = np.exp(1j * phase_traj)
    modulated = analytic * modulation
    out = np.real(modulated).astype(np.float32)

    # apply amplitude envelope
    out *= amp_curve.astype(np.float32)

    # normalize
    maxv = np.max(np.abs(out)) if out.size else 1.0
    if maxv > 0:
        out = out / maxv * 0.98

    return out


# Front-facing functions used by app.py
def apply_doppler_to_audio_fixed(original_audio, freq_ratios, amplitudes):
    """
    Default method: time-domain warping (fast and produces clear sweeps).
    """
    return apply_true_doppler_shift(original_audio, freq_ratios, amplitudes)


def apply_doppler_to_audio_fixed_alternative(original_audio, freq_ratios, amplitudes):
    """
    Alternative method: spectral warping (slower; good visual control).
    """
    return apply_spectral_doppler_shift(original_audio, freq_ratios, amplitudes)


def apply_doppler_to_audio_fixed_advanced(original_audio, freq_ratios, amplitudes):
    """
    Advanced method: phase modulation (strong, clean sweeps).
    """
    return apply_phase_modulation_doppler(original_audio, freq_ratios, amplitudes)


# ---------------------------
# Utilities
# ---------------------------
def normalize_amplitudes(amplitudes):
    """Normalize amplitudes list/array to [0,1] range and return list."""
    if amplitudes is None or len(amplitudes) == 0:
        return amplitudes
    arr = np.asarray(amplitudes, dtype=np.float32)
    maxv = np.max(np.abs(arr))
    if maxv <= 0:
        return arr.tolist()
    arr = arr / maxv
    return arr.tolist()


def save_audio(audio_data, output_path):
    """
    Save audio_data (1D numpy array) to output_path using SR sample rate.
    Returns duration in seconds.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = np.asarray(audio_data, dtype=np.float32)
    # prevent clipping
    maxv = np.max(np.abs(data)) if data.size else 1.0
    if maxv > 0:
        data = data / maxv * 0.98
    sf.write(output_path, data, SR, subtype='PCM_16')
    return float(len(data) / SR)


# ---------------------------
# Debug / analysis helpers
# ---------------------------
def analyze_doppler_effect(original_audio, processed_audio, freq_ratios):
    """
    Simple analysis that prints a comparison of expected vs actual dominant-frequency ratios.
    Useful during debugging.
    """
    original = np.asarray(original_audio, dtype=np.float32)
    processed = np.asarray(processed_audio, dtype=np.float32)
    n_fft = 2048
    hop_length = 512
    orig_stft = librosa.stft(original, n_fft=n_fft, hop_length=hop_length)
    proc_stft = librosa.stft(processed, n_fft=n_fft, hop_length=hop_length)
    orig_mag = np.abs(orig_stft)
    proc_mag = np.abs(proc_stft)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)

    orig_dom = []
    proc_dom = []
    frames = min(orig_mag.shape[1], proc_mag.shape[1])
    for f in range(frames):
        orig_dom.append(freqs[np.argmax(orig_mag[:, f])])
        proc_dom.append(freqs[np.argmax(proc_mag[:, f])])

    orig_dom = np.array(orig_dom)
    proc_dom = np.array(proc_dom)
    # Avoid division by zero
    ratio_actual = proc_dom / (orig_dom + 1e-10)
    print("Expected freq ratio range:", np.min(freq_ratios), np.max(freq_ratios))
    print("Actual dominant freq ratio range:", np.min(ratio_actual), np.max(ratio_actual))
    return orig_dom, proc_dom, ratio_actual


# ---------------------------
# Quick test (when run directly)
# ---------------------------
if __name__ == "__main__":
    print("audio_utils.py self-test")
    # Generate a short test signal
    duration = 3.0
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    test_audio = (np.sin(2*np.pi*440*t) + 0.7*np.sin(2*np.pi*880*t)).astype(np.float32)
    # sample freq ratios (simple pass-by)
    n_points = 80
    t_curve = np.linspace(0, 1, n_points)
    freq_ratios = 1.2 * np.exp(-((t_curve - 0.5)/0.18)**2) + 0.8
    amplitudes = 1.0 / (1.0 + 3.0 * (t_curve - 0.5)**2)
    out = apply_doppler_to_audio_fixed(test_audio, freq_ratios.tolist(), amplitudes.tolist())
    analyze_doppler_effect(test_audio, out, freq_ratios)
    print("Test done. Output length:", len(out), "samples.")
