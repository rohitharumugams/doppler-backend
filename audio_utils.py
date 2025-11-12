import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from scipy.interpolate import interp1d

SOUND_DURATION = 5  # Default seconds - will be overridden by user input
SR = 22050  # Sample rate

def load_original_audio(audio_type='horn', duration=5):
    """Load the original audio based on vehicle type and duration"""
    audio_files = {
        'car': 'static/horn.mp3',
        'train': 'static/train.mp3', 
        'flight': 'static/flight.mp3'
    }
    
    filename = audio_files.get(audio_type, 'static/horn.mp3')
    
    try:
        # Load without duration limit first to get the full file
        y, original_sr = librosa.load(filename, sr=SR, mono=True)
        original_duration = len(y) / SR
        
        print(f"Loaded {audio_type} audio from {filename}")
        print(f"Original duration: {original_duration:.2f}s, Requested: {duration}s")
        
        if original_duration >= duration:
            # If original is longer than requested, just trim it
            target_samples = int(SR * duration)
            y = y[:target_samples]
            print(f"Trimmed to {duration}s")
        else:
            # If original is shorter, repeat and stitch with overlaps
            y = extend_audio_with_overlap(y, duration, SR)
            print(f"Extended to {duration}s with seamless overlaps")
        
        return y
        
    except Exception as e:
        print(f"Could not load {filename}: {e}")
        print(f"Generating fallback {audio_type} sound (duration: {duration}s)...")
        
        # Generate fallback sounds based on type
        t = np.linspace(0, duration, int(SR * duration))
        
        if audio_type == 'car':
            # Car horn - multiple harmonics
            audio = (np.sin(2 * np.pi * 440 * t) + 
                    0.7 * np.sin(2 * np.pi * 880 * t) + 
                    0.4 * np.sin(2 * np.pi * 1320 * t) +
                    0.2 * np.sin(2 * np.pi * 660 * t))
            envelope = np.exp(-t * 0.3) * 0.5 + 0.5
            
        elif audio_type == 'train':
            # Train - lower frequencies with rhythm
            audio = (np.sin(2 * np.pi * 220 * t) + 
                    0.8 * np.sin(2 * np.pi * 110 * t) + 
                    0.6 * np.sin(2 * np.pi * 330 * t) +
                    0.3 * np.sin(2 * np.pi * 440 * t))
            # Add rhythmic component for train-like sound
            rhythm = 1 + 0.3 * np.sin(2 * np.pi * 8 * t)
            audio *= rhythm
            envelope = 0.8 + 0.2 * np.sin(2 * np.pi * 2 * t)
            
        elif audio_type == 'flight':
            # Flight - higher frequencies with turbine-like sound
            audio = (0.6 * np.sin(2 * np.pi * 800 * t) + 
                    0.8 * np.sin(2 * np.pi * 1200 * t) + 
                    0.4 * np.sin(2 * np.pi * 1600 * t) +
                    0.3 * np.sin(2 * np.pi * 400 * t))
            # Add turbine-like modulation
            turbine = 1 + 0.2 * np.sin(2 * np.pi * 15 * t)
            audio *= turbine
            envelope = 0.9 + 0.1 * np.sin(2 * np.pi * 3 * t)
        else:
            # Default fallback (same as car)
            audio = (np.sin(2 * np.pi * 440 * t) + 
                    0.7 * np.sin(2 * np.pi * 880 * t) + 
                    0.4 * np.sin(2 * np.pi * 1320 * t) +
                    0.2 * np.sin(2 * np.pi * 660 * t))
            envelope = np.exp(-t * 0.3) * 0.5 + 0.5
        
        return audio * envelope


def extend_audio_with_overlap(original_audio, target_duration, sample_rate):
    """
    Extend audio to target duration by repeating with smooth overlaps
    
    Args:
        original_audio: numpy array of audio samples
        target_duration: desired duration in seconds
        sample_rate: audio sample rate
    
    Returns:
        numpy array: extended audio with seamless transitions
    """
    original_length = len(original_audio)
    original_duration = original_length / sample_rate
    target_length = int(sample_rate * target_duration)
    
    if original_length >= target_length:
        return original_audio[:target_length]
    
    # Calculate overlap parameters
    overlap_duration = 0.1  # 100ms overlap for smooth transitions
    overlap_samples = int(sample_rate * overlap_duration)
    overlap_samples = min(overlap_samples, original_length // 4)  # Don't overlap more than 25% of original
    
    print(f"  Using {overlap_samples} samples ({overlap_samples/sample_rate*1000:.0f}ms) overlap")
    
    # Create extended audio array
    extended_audio = np.zeros(target_length)
    
    # Copy first iteration completely
    extended_audio[:original_length] = original_audio
    current_position = original_length
    
    # Add subsequent iterations with crossfade overlaps
    iteration = 1
    while current_position < target_length:
        # Calculate how much we need and can fit
        remaining_samples = target_length - current_position
        
        if remaining_samples <= overlap_samples:
            # Last piece - just fade out what we have
            break
            
        # Start position for this iteration (with overlap)
        start_pos = current_position - overlap_samples
        
        # How much of the original audio to use
        samples_to_use = min(original_length, remaining_samples + overlap_samples)
        end_pos = start_pos + samples_to_use
        
        if end_pos > target_length:
            samples_to_use = target_length - start_pos
            end_pos = target_length
        
        # Create crossfade window for overlap region
        if overlap_samples > 0:
            # Fade out existing audio
            fade_out = np.linspace(1, 0, overlap_samples)
            # Fade in new audio
            fade_in = np.linspace(0, 1, overlap_samples)
            
            # Apply crossfade in overlap region
            overlap_end = start_pos + overlap_samples
            
            # Fade out existing content
            extended_audio[start_pos:overlap_end] *= fade_out
            
            # Add faded-in new content
            new_audio_overlap = original_audio[:overlap_samples] * fade_in
            extended_audio[start_pos:overlap_end] += new_audio_overlap
            
            # Add rest of new audio (non-overlapping part)
            if samples_to_use > overlap_samples:
                non_overlap_samples = samples_to_use - overlap_samples
                extended_audio[overlap_end:end_pos] = original_audio[overlap_samples:overlap_samples + non_overlap_samples]
        else:
            # No overlap - just concatenate
            extended_audio[start_pos:end_pos] = original_audio[:samples_to_use]
        
        # Update position for next iteration
        current_position = end_pos
        iteration += 1
    
    print(f"  Extended audio using {iteration} iterations with smooth crossfades")
    
    # Apply gentle fade out at the very end to avoid clicks
    fade_length = min(int(sample_rate * 0.05), target_length // 20)  # 50ms or 5% of duration
    if fade_length > 0:
        fade_out_final = np.linspace(1, 0, fade_length)
        extended_audio[-fade_length:] *= fade_out_final
    
    return extended_audio

# Keep backward compatibility
def load_original_horn():
    """Backward compatibility function"""
    return load_original_audio('car', 5)

def apply_true_doppler_shift(original_audio, freq_ratios, amplitudes):
    """
    Apply TRUE Doppler shift that will show proper frequency sweeps in spectrogram.
    This creates strong, visible frequency modulation over time.
    """
    target_length = len(original_audio)
    
    # Create time axes
    time_samples = np.arange(target_length)
    curve_time = np.linspace(0, target_length-1, len(freq_ratios))
    
    # Interpolate frequency ratios and amplitudes to match audio length
    freq_interp = interp1d(curve_time, freq_ratios, kind='cubic', bounds_error=False, fill_value='extrapolate')
    amp_interp = interp1d(curve_time, amplitudes, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    freq_curve = freq_interp(time_samples)
    amp_curve = amp_interp(time_samples)
    
    # Apply minimal smoothing to preserve sharp frequency changes
    if len(freq_curve) > 50:
        window_size = min(11, len(freq_curve) // 100 * 2 + 1)  # Much smaller window
        freq_curve = signal.savgol_filter(freq_curve, window_size, 1)  # Lower order
        amp_curve = signal.savgol_filter(amp_curve, window_size, 1)
    
    print(f"Applying TRUE Doppler shift with STRONG frequency modulation")
    print(f"Frequency ratio range: {np.min(freq_curve):.3f} to {np.max(freq_curve):.3f}")
    print(f"Frequency variation: {np.std(freq_curve):.3f}")
    
    # METHOD: Direct instantaneous frequency modulation
    # This will create visible sweeps by modulating the instantaneous frequency
    
    # Create the time-varying frequency modulation
    # Convert frequency ratios to instantaneous frequency multipliers
    dt = 1.0 / SR
    
    # Calculate instantaneous phase increments
    # This directly controls how fast we move through the original audio
    phase_increments = freq_curve * dt * SR
    
    # Calculate cumulative phase (sample positions in original audio)
    cumulative_phase = np.cumsum(phase_increments)
    
    # Normalize to stay within audio bounds
    max_phase = target_length - 1
    if cumulative_phase[-1] > max_phase:
        cumulative_phase = cumulative_phase * max_phase / cumulative_phase[-1]
    
    # Sample original audio at these phase positions using interpolation
    output = np.interp(cumulative_phase, np.arange(target_length), original_audio)
    
    # Apply amplitude modulation
    output *= amp_curve
    
    # Verify the effect strength
    print(f"Phase range: 0 to {cumulative_phase[-1]:.1f} (should be close to {max_phase})")
    print(f"Expected strong frequency sweeps in spectrogram")
    
    return output

def apply_spectral_doppler_shift(original_audio, freq_ratios, amplitudes):
    """
    Enhanced spectral method with stronger frequency shifts for visible spectrogram sweeps.
    """
    target_length = len(original_audio)
    
    # Create time axes
    time_samples = np.arange(target_length)
    curve_time = np.linspace(0, target_length-1, len(freq_ratios))
    
    # Interpolate frequency ratios and amplitudes
    freq_interp = interp1d(curve_time, freq_ratios, kind='cubic', bounds_error=False, fill_value='extrapolate')
    amp_interp = interp1d(curve_time, amplitudes, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    freq_curve = freq_interp(time_samples)
    amp_curve = amp_interp(time_samples)
    
    # STFT parameters for good time-frequency resolution
    n_fft = 1024  # Smaller for better time resolution
    hop_length = 256  # Smaller hop for smoother frequency transitions
    
    print(f"Applying ENHANCED SPECTRAL Doppler shift")
    print(f"Frequency ratio range: {np.min(freq_curve):.3f} to {np.max(freq_curve):.3f}")
    print(f"Using smaller STFT windows for better frequency sweep visibility")
    
    # Compute STFT of original audio
    stft = librosa.stft(original_audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Get frequency bins
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    
    # Create output STFT
    output_stft = np.zeros_like(stft, dtype=complex)
    
    # Process each time frame with stronger frequency shifting
    n_frames = stft.shape[1]
    for frame_idx in range(n_frames):
        # Get time position for this frame
        time_sample = int(frame_idx * hop_length)
        
        # Get frequency ratio at this time
        if time_sample < len(freq_curve):
            freq_ratio = freq_curve[time_sample]
            amplitude = amp_curve[time_sample]
        else:
            freq_ratio = freq_curve[-1]
            amplitude = amp_curve[-1]
        
        # Apply stronger frequency shifting
        current_magnitude = magnitude[:, frame_idx]
        current_phase = phase[:, frame_idx]
        
        # Create new frequency mapping with wider range
        new_magnitude = np.zeros_like(current_magnitude)
        new_phase = np.zeros_like(current_phase)
        
        # Shift each frequency bin
        for freq_idx in range(len(freqs)):
            if freq_idx == 0:  # Skip DC component
                new_magnitude[0] = current_magnitude[0]
                new_phase[0] = current_phase[0]
                continue
                
            old_freq = freqs[freq_idx]
            new_freq = old_freq * freq_ratio
            
            # Find target frequency bin(s)
            if new_freq > 0 and new_freq < freqs[-1]:
                # Use linear interpolation for smoother frequency mapping
                target_idx = np.interp(new_freq, freqs, np.arange(len(freqs)))
                
                # Distribute energy between adjacent bins
                lower_idx = int(np.floor(target_idx))
                upper_idx = int(np.ceil(target_idx))
                
                if lower_idx < len(new_magnitude) and upper_idx < len(new_magnitude):
                    weight = target_idx - lower_idx
                    
                    # Distribute magnitude and preserve phase relationships
                    new_magnitude[lower_idx] += current_magnitude[freq_idx] * (1 - weight)
                    new_magnitude[upper_idx] += current_magnitude[freq_idx] * weight
                    
                    # Preserve phase for coherent reconstruction
                    new_phase[lower_idx] = current_phase[freq_idx]
                    new_phase[upper_idx] = current_phase[freq_idx]
        
        # Apply amplitude modulation and store result
        output_stft[:, frame_idx] = new_magnitude * np.exp(1j * new_phase) * amplitude
    
    # Convert back to time domain
    output = librosa.istft(output_stft, hop_length=hop_length, length=target_length)
    
    print(f"Spectral processing complete - should show clear frequency sweeps")
    
    return output

def apply_phase_modulation_doppler(original_audio, freq_ratios, amplitudes):
    """
    Enhanced phase modulation method for maximum frequency sweep visibility.
    This creates the strongest and clearest frequency sweeps in spectrograms.
    
    FIXED: Removed artificial 2x amplification to maintain realistic Doppler effect.
    """
    target_length = len(original_audio)
    
    # Create time axes
    time_samples = np.arange(target_length)
    curve_time = np.linspace(0, target_length-1, len(freq_ratios))
    
    # Interpolate frequency ratios and amplitudes
    freq_interp = interp1d(curve_time, freq_ratios, kind='cubic', bounds_error=False, fill_value='extrapolate')
    amp_interp = interp1d(curve_time, amplitudes, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    freq_curve = freq_interp(time_samples)
    amp_curve = amp_interp(time_samples)
    
    # NO smoothing to preserve sharp frequency transitions
    print(f"Applying ENHANCED PHASE MODULATION Doppler shift")
    print(f"Frequency ratio range: {np.min(freq_curve):.3f} to {np.max(freq_curve):.3f}")
    print(f"Creating maximum frequency sweep visibility via phase modulation")
    
    # Enhanced phase modulation approach
    # Create instantaneous frequency modulation that's highly visible
    
    # Calculate instantaneous frequency deviation from unity
    freq_deviation = freq_curve - 1.0
    
    # FIXED: Removed artificial 2x scaling to maintain realistic Doppler effect
    # Original line was: freq_deviation *= 2.0  # Make frequency changes more dramatic
    # Now we keep the physically accurate frequency deviation
    
    # Calculate cumulative phase modulation
    dt = 1.0 / SR
    
    # Create phase trajectory - this directly controls instantaneous frequency
    phase_trajectory = np.zeros(target_length)
    for i in range(1, target_length):
        # Instantaneous frequency = 1 + deviation
        instantaneous_freq = 1.0 + freq_deviation[i]
        # Add to cumulative phase
        phase_trajectory[i] = phase_trajectory[i-1] + 2 * np.pi * instantaneous_freq * dt
    
    # Create complex modulation signal
    modulation_signal = np.exp(1j * phase_trajectory)
    
    # Convert original audio to complex (analytic) signal
    analytic_signal = signal.hilbert(original_audio)
    
    # Apply frequency modulation by multiplying with modulation signal
    modulated_signal = analytic_signal * modulation_signal
    
    # Take real part to get final audio
    output = np.real(modulated_signal)
    
    # Apply amplitude modulation
    output *= amp_curve
    
    # Verify modulation strength
    freq_variation = np.std(freq_deviation)
    print(f"Frequency deviation std: {freq_variation:.3f}")
    print(f"Phase trajectory range: {np.min(phase_trajectory):.1f} to {np.max(phase_trajectory):.1f}")
    print(f"Should produce visible frequency sweeps with physically accurate Doppler effect")
    
    return output

def apply_doppler_to_audio_fixed(original_audio, freq_ratios, amplitudes):
    """
    Main function that applies TRUE Doppler shift with visible spectrogram sweeps
    """
    print("=" * 60)
    print("APPLYING TRUE DOPPLER SHIFT FOR SPECTROGRAM VISIBILITY")
    print("=" * 60)
    
    # Use the variable resampling method as primary
    result = apply_true_doppler_shift(original_audio, freq_ratios, amplitudes)
    
    return result

def apply_doppler_to_audio_fixed_alternative(original_audio, freq_ratios, amplitudes):
    """
    Alternative using spectral method
    """
    print("=" * 60)
    print("APPLYING SPECTRAL DOPPLER SHIFT")
    print("=" * 60)
    
    result = apply_spectral_doppler_shift(original_audio, freq_ratios, amplitudes)
    
    return result

def apply_doppler_to_audio_fixed_advanced(original_audio, freq_ratios, amplitudes):
    """
    Advanced method using phase modulation
    """
    print("=" * 60)
    print("APPLYING PHASE MODULATION DOPPLER SHIFT")
    print("=" * 60)
    
    result = apply_phase_modulation_doppler(original_audio, freq_ratios, amplitudes)
    
    return result

def normalize_amplitudes(amplitudes):
    """Normalize amplitudes to [0, 1] range"""
    if amplitudes:
        max_amp = max(amplitudes)
        if max_amp > 0:
            return [a / max_amp for a in amplitudes]
    return amplitudes

def save_audio(audio_data, output_path):
    """Save audio data to file"""
    # Normalize to prevent clipping
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val * 0.8
    
    sf.write(output_path, audio_data, SR)
    return len(audio_data) / SR

def analyze_doppler_effect(original_audio, processed_audio, freq_ratios):
    """
    Analyze the Doppler effect to verify it's working correctly
    """
    print("\n" + "="*50)
    print("DOPPLER EFFECT ANALYSIS")
    print("="*50)
    
    # Compute spectrograms
    n_fft = 2048
    hop_length = 512
    
    orig_stft = librosa.stft(original_audio, n_fft=n_fft, hop_length=hop_length)
    proc_stft = librosa.stft(processed_audio, n_fft=n_fft, hop_length=hop_length)
    
    orig_mag = np.abs(orig_stft)
    proc_mag = np.abs(proc_stft)
    
    # Find dominant frequency over time
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    
    orig_dominant_freqs = []
    proc_dominant_freqs = []
    
    for frame in range(orig_mag.shape[1]):
        # Original audio dominant frequency
        orig_peak_idx = np.argmax(orig_mag[:, frame])
        orig_dominant_freqs.append(freqs[orig_peak_idx])
        
        # Processed audio dominant frequency  
        proc_peak_idx = np.argmax(proc_mag[:, frame])
        proc_dominant_freqs.append(freqs[proc_peak_idx])
    
    orig_dominant_freqs = np.array(orig_dominant_freqs)
    proc_dominant_freqs = np.array(proc_dominant_freqs)
    
    # Calculate frequency ratio over time
    actual_freq_ratios = proc_dominant_freqs / (orig_dominant_freqs + 1e-10)
    
    print(f"Expected frequency ratio range: {np.min(freq_ratios):.3f} to {np.max(freq_ratios):.3f}")
    print(f"Actual frequency ratio range: {np.min(actual_freq_ratios):.3f} to {np.max(actual_freq_ratios):.3f}")
    
    # Check if we see the expected frequency sweep
    freq_variation = np.std(actual_freq_ratios)
    print(f"Frequency variation (std): {freq_variation:.3f}")
    
    if freq_variation > 0.05:
        print("✅ GOOD: Significant frequency variation detected - should see sweeps in spectrogram")
    else:
        print("⚠ PROBLEM: Little frequency variation - spectrogram may appear flat")
    
    return orig_dominant_freqs, proc_dominant_freqs, actual_freq_ratios

def test_doppler_with_analysis():
    """
    Test function with detailed analysis
    """
    print("Testing Doppler shift with spectrogram analysis...")
    
    # Create test audio
    duration = 3.0
    t = np.linspace(0, duration, int(SR * duration))
    
    # Multi-harmonic test signal (like a horn)
    test_audio = (np.sin(2 * np.pi * 440 * t) + 
                  0.7 * np.sin(2 * np.pi * 880 * t) + 
                  0.4 * np.sin(2 * np.pi * 1320 * t))
    
    # Create realistic Doppler frequency ratios (approaching then receding)
    num_points = 100
    t_curve = np.linspace(0, 1, num_points)
    
    # Simulate vehicle passing by (high freq -> low freq)
    freq_ratios = 1.3 * np.exp(-((t_curve - 0.5) / 0.2)**2) + 0.7
    amplitudes = 1.0 / (((t_curve - 0.5) * 40)**2 + 1)
    
    print(f"Test frequency ratios: {np.min(freq_ratios):.3f} to {np.max(freq_ratios):.3f}")
    
    # Apply Doppler effect
    result = apply_doppler_to_audio_fixed(test_audio, freq_ratios.tolist(), amplitudes.tolist())
    
    # Analyze results
    analyze_doppler_effect(test_audio, result, freq_ratios)
    
    return test_audio, result, freq_ratios

if __name__ == "__main__":
    # Run test with analysis
    test_doppler_with_analysis()