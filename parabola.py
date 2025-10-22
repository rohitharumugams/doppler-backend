import numpy as np

def calculate_parabola_doppler(speed, a, h, duration=5, num_points=None):
    """
    Calculate Doppler effect for parabolic motion based on speed and duration
    
    Args:
        speed: horizontal speed in m/s
        a: parabola coefficient (curvature) 
        h: vertical offset (height at x=0) in meters
        duration: total audio duration in seconds
        num_points: number of calculation points (defaults to audio sample rate equivalent)
    
    Returns:
        tuple: (freq_ratios, amplitudes)
    """
    c = 343.0  # Speed of sound in m/s
    
    # Use full audio resolution for smooth Doppler effect
    if num_points is None:
        num_points = int(duration * 22050 / 100)  # Reasonable number of points
    
    # Calculate total distance traveled during duration
    total_distance = speed * duration
    
    # Time array centered around t=0 (closest approach in middle)
    t_vals = np.linspace(-duration/2, duration/2, num_points)
    
    print(f"Parabolic motion:")
    print(f"  Speed: {speed} m/s")
    print(f"  Duration: {duration} s")
    print(f"  Total distance: {total_distance:.1f} m")
    print(f"  Curvature (a): {a}")
    print(f"  Height at center (h): {h} m")
    print(f"  Audio samples: {num_points}")
    
    # Vehicle position along parabola y = ax² + h
    x = speed * t_vals  # Horizontal position based on speed and time
    y = a * x**2 + h   # Vertical position (parabola equation)
    
    # Vehicle velocity components
    vx_arr = np.full_like(x, speed)  # Constant horizontal velocity
    vy = 2 * a * x * speed  # Vertical velocity: d/dt(a(speed*t)²) = 2a*speed²*t
    
    # Line-of-sight distance from observer (at origin) to vehicle
    r = np.sqrt(x**2 + y**2)
    
    # Prevent division by zero for very close approaches
    r = np.maximum(r, 0.1)
    
    # Radial velocity component (positive = approaching observer)
    # vr = (v⃗ · r⃗) / |r⃗| where v⃗ = (vx, vy) and r⃗ = (x, y)
    vr = (vx_arr * x + vy * y) / r
    
    # Apply velocity smoothing for more natural sound transitions
    if len(vr) > 10:
        from scipy.signal import savgol_filter
        window_size = min(21, len(vr) // 20 * 2 + 1)  # Odd number for savgol
        if window_size >= 3:
            try:
                vr = savgol_filter(vr, window_size, 2)
            except:
                pass  # Use original if smoothing fails
    
    # Doppler frequency ratio: f'/f = c/(c - vr)
    # When vr > 0 (approaching): frequency increases
    # When vr < 0 (receding): frequency decreases
    # Limit radial velocity to prevent extreme frequency shifts
    vr_limited = np.clip(vr, -c*0.3, c*0.3)  # Limit to ±30% of sound speed
    freq_ratios = c / (c - vr_limited)
    
    # FIXED: More physical amplitude calculation (inverse distance law)
    base_amplitude = 1.0 / (r + 1.0)  # Simple inverse distance
    
    # Add subtle directional effects for parabolic motion
    velocity_magnitude = np.sqrt(vx_arr**2 + vy**2)
    angle_factor = np.abs(vr) / (velocity_magnitude + 1e-6)  # Normalized radial component
    directional_factor = 1.0 + 0.1 * angle_factor  # Slight boost when moving directly toward/away
    
    # Combine amplitude factors
    amplitudes = base_amplitude * directional_factor
    
    # Apply gentle amplitude envelope to avoid abrupt start/end
    envelope_length = min(num_points // 10, int(0.2 * 22050))  # 0.2 second fade
    if envelope_length > 0:
        fade_in = np.linspace(0, 1, envelope_length)
        fade_out = np.linspace(1, 0, envelope_length)
        
        amplitudes[:envelope_length] *= fade_in
        amplitudes[-envelope_length:] *= fade_out
    
    # Find closest approach for debugging and verification
    min_distance_idx = np.argmin(r)
    min_distance = r[min_distance_idx]
    closest_t = t_vals[min_distance_idx]
    closest_x = x[min_distance_idx]
    closest_y = y[min_distance_idx]
    max_radial_vel = np.max(np.abs(vr))
    
    print(f"  Closest approach: {min_distance:.2f} m at t={closest_t:.3f} s")
    print(f"  Closest point: ({closest_x:.2f}, {closest_y:.2f}) m")
    print(f"  Max radial velocity: {max_radial_vel:.2f} m/s ({max_radial_vel/c*100:.1f}% of c)")
    print(f"  Frequency ratio range: {np.min(freq_ratios):.3f} to {np.max(freq_ratios):.3f}")
    print(f"  Start position: ({x[0]:.1f}, {y[0]:.1f}) m")
    print(f"  End position: ({x[-1]:.1f}, {y[-1]:.1f}) m")
    
    # Verify parabola equation at closest approach
    if abs(closest_x) < 1e-6:  # Should be very close to x=0 for symmetric parabola
        expected_y = h
        print(f"  Vertex verification: y={closest_y:.2f} (expected {expected_y:.2f})")
    
    return freq_ratios.tolist(), amplitudes.tolist()