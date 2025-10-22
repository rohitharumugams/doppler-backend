import numpy as np

def calculate_bezier_doppler(speed, x0, x1, x2, x3, y0, y1, y2, y3, duration=5, num_points=None):
    """
    Calculate Doppler effect for Bezier curve motion based on speed and duration
    
    NOTE: The speed parameter controls traversal rate, but due to Bezier curve geometry,
    the actual geometric speed varies along the curve. This implementation provides
    an approximation by scaling parametric velocity to match the desired speed.
    For true constant speed, arc-length reparameterization would be needed (complex).
    
    The sound source moves along a cubic Bezier curve defined by 4 control points,
    but the traversal speed is controlled by the speed parameter.
    
    Args:
        speed: vehicle speed in m/s (controls how fast it moves along the curve)
        x0, y0: Start point coordinates (m)
        x1, y1: First control point coordinates (m) 
        x2, y2: Second control point coordinates (m)
        x3, y3: End point coordinates (m)
        duration: total audio duration in seconds
        num_points: number of calculation points (defaults to audio sample rate)
    
    Returns:
        tuple: (freq_ratios, amplitudes)
    """
    c = 343.0  # Speed of sound in m/s
    
    # Use full audio resolution for smooth Doppler effect
    if num_points is None:
        num_points = int(duration * 22050 / 100)  # Reasonable number of points
    
    # Calculate the total arc length of the Bezier curve
    curve_length = calculate_bezier_arc_length(x0, x1, x2, x3, y0, y1, y2, y3)
    
    # Calculate how much of the curve we traverse based on speed and duration
    total_distance = speed * duration
    
    # Determine what portion of the curve we actually travel
    if total_distance >= curve_length:
        # We travel the entire curve
        t_start = 0.0
        t_end = 1.0
        actual_distance = curve_length
        print(f"  NOTE: Speed allows traveling entire curve ({curve_length:.1f}m)")
    else:
        # We travel only part of the curve, centered around the middle
        curve_fraction = total_distance / curve_length
        t_center = 0.5  # Center of curve
        t_half_span = curve_fraction / 2
        t_start = max(0.0, t_center - t_half_span)
        t_end = min(1.0, t_center + t_half_span)
        actual_distance = total_distance
        print(f"  Traveling {curve_fraction*100:.1f}% of curve (center portion)")
    
    # Parameter t goes from t_start to t_end over the duration
    t_vals = np.linspace(t_start, t_end, num_points)
    
    print(f"Bezier curve motion:")
    print(f"  Speed: {speed} m/s")
    print(f"  Duration: {duration} s")
    print(f"  Total curve length: {curve_length:.1f} m")
    print(f"  Distance at this speed: {total_distance:.1f} m")
    print(f"  Actual distance traveled: {actual_distance:.1f} m")
    print(f"  P0: ({x0:.1f}, {y0:.1f}) m")
    print(f"  P1: ({x1:.1f}, {y1:.1f}) m") 
    print(f"  P2: ({x2:.1f}, {y2:.1f}) m")
    print(f"  P3: ({x3:.1f}, {y3:.1f}) m")
    print(f"  t range: {t_start:.3f} to {t_end:.3f}")
    print(f"  Audio samples: {num_points}")
    
    # Calculate position and velocity along Bezier curve
    positions = []
    velocities = []
    
    for t in t_vals:
        # Bezier curve position: P(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
        pos_x = bezier_point(t, x0, x1, x2, x3)
        pos_y = bezier_point(t, y0, y1, y2, y3)
        positions.append((pos_x, pos_y))
        
        # Bezier curve velocity: P'(t) = derivative of position
        # Scale by actual speed: velocity magnitude should match our speed parameter
        # NOTE: This is an approximation - true constant speed requires arc-length parameterization
        vel_x_param = bezier_velocity(t, x0, x1, x2, x3)  # Parametric velocity
        vel_y_param = bezier_velocity(t, y0, y1, y2, y3)
        
        # Calculate parametric speed at this point
        param_speed = np.sqrt(vel_x_param**2 + vel_y_param**2)
        
        # Scale to match actual speed
        if param_speed > 1e-6:  # Avoid division by zero
            scale_factor = speed / param_speed
            vel_x = vel_x_param * scale_factor
            vel_y = vel_y_param * scale_factor
        else:
            vel_x = speed  # Fallback
            vel_y = 0
            
        velocities.append((vel_x, vel_y))
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    # Calculate distances from observer (at origin)
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    distances = np.sqrt(x_coords**2 + y_coords**2)
    
    # Prevent division by zero for very close approaches
    distances = np.maximum(distances, 0.1)
    
    # Calculate radial velocity components (positive = approaching observer)
    # vr = (v⃗ · r⃗) / |r⃗| where v⃗ = velocity vector and r⃗ = position vector
    vx_arr = velocities[:, 0]
    vy_arr = velocities[:, 1]
    radial_velocities = (vx_arr * x_coords + vy_arr * y_coords) / distances
    
    # Apply velocity smoothing for more natural sound transitions
    if len(radial_velocities) > 10:
        window_size = min(21, len(radial_velocities) // 20 * 2 + 1)  # Odd number for savgol
        if window_size >= 3:
            from scipy.signal import savgol_filter
            try:
                radial_velocities = savgol_filter(radial_velocities, window_size, 2)
            except:
                print("Warning: Smoothing failed, using original velocities")
                pass  # Use original if smoothing fails
    
    # Doppler frequency ratio: f'/f = c/(c - vr)
    # When vr > 0 (approaching): frequency increases
    # When vr < 0 (receding): frequency decreases
    # Limit radial velocity to prevent extreme frequency shifts
    vr_limited = np.clip(radial_velocities, -c*0.3, c*0.3)  # Limit to ±30% of sound speed
    freq_ratios = c / (c - vr_limited)
    
    # FIXED: More physical amplitude calculation (inverse distance law)
    base_amplitude = 1.0 / (distances + 1.0)  # Simple inverse distance
    
    # Add subtle directional effects for Bezier curve motion
    # Sound slightly different when vehicle is moving at different angles
    velocity_magnitudes = np.sqrt(vx_arr**2 + vy_arr**2)
    # Avoid division by zero
    velocity_magnitudes = np.maximum(velocity_magnitudes, 1e-6)
    angle_factors = np.abs(radial_velocities) / velocity_magnitudes  # Normalized radial component
    directional_factors = 1.0 + 0.1 * angle_factors  # Slight boost when moving directly toward/away
    
    # Combine amplitude factors
    amplitudes = base_amplitude * directional_factors
    
    # Apply gentle amplitude envelope to avoid abrupt start/end
    envelope_length = min(num_points // 10, int(0.2 * 22050))  # 0.2 second fade
    if envelope_length > 0:
        fade_in = np.linspace(0, 1, envelope_length)
        fade_out = np.linspace(1, 0, envelope_length)
        
        amplitudes[:envelope_length] *= fade_in
        amplitudes[-envelope_length:] *= fade_out
    
    # Find closest approach for debugging and verification
    min_distance_idx = np.argmin(distances)
    min_distance = distances[min_distance_idx]
    closest_t = t_vals[min_distance_idx]
    closest_pos = positions[min_distance_idx]
    max_radial_vel = np.max(np.abs(radial_velocities))
    
    print(f"  Closest approach: {min_distance:.2f} m at t={closest_t:.3f}")
    print(f"  Closest point: ({closest_pos[0]:.2f}, {closest_pos[1]:.2f}) m")
    print(f"  Max radial velocity: {max_radial_vel:.2f} m/s ({max_radial_vel/c*100:.1f}% of c)")
    print(f"  Frequency ratio range: {np.min(freq_ratios):.3f} to {np.max(freq_ratios):.3f}")
    print(f"  Amplitude range: {np.min(amplitudes):.4f} to {np.max(amplitudes):.4f}")
    
    # Calculate curve statistics
    avg_speed = actual_distance / duration
    max_speed = np.max(velocity_magnitudes)
    
    print(f"  Curve length: {curve_length:.1f} m")
    print(f"  Average speed: {avg_speed:.1f} m/s")
    print(f"  Maximum speed: {max_speed:.1f} m/s")
    print(f"  Start position: ({x0:.1f}, {y0:.1f}) m")
    print(f"  End position: ({x3:.1f}, {y3:.1f}) m")
    
    return freq_ratios.tolist(), amplitudes.tolist()


def bezier_point(t, p0, p1, p2, p3):
    """
    Calculate a point on a cubic Bezier curve at parameter t
    
    Args:
        t: parameter (0 to 1)
        p0, p1, p2, p3: control point coordinates
    
    Returns:
        float: coordinate value at parameter t
    """
    return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def bezier_velocity(t, p0, p1, p2, p3):
    """
    Calculate velocity (derivative) of a cubic Bezier curve at parameter t
    
    Args:
        t: parameter (0 to 1)
        p0, p1, p2, p3: control point coordinates
    
    Returns:
        float: velocity component at parameter t
    """
    return 3 * (1 - t)**2 * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * t**2 * (p3 - p2)


def calculate_bezier_arc_length(x0, x1, x2, x3, y0, y1, y2, y3, num_segments=1000):
    """
    Calculate approximate arc length of Bezier curve using numerical integration
    
    Returns:
        float: arc length in meters
    """
    t_vals = np.linspace(0, 1, num_segments + 1)
    total_length = 0.0
    
    for i in range(num_segments):
        t1, t2 = t_vals[i], t_vals[i + 1]
        
        # Calculate positions at t1 and t2
        x1_pos = bezier_point(t1, x0, x1, x2, x3)
        y1_pos = bezier_point(t1, y0, y1, y2, y3)
        x2_pos = bezier_point(t2, x0, x1, x2, x3)
        y2_pos = bezier_point(t2, y0, y1, y2, y3)
        
        # Add segment length
        segment_length = np.sqrt((x2_pos - x1_pos)**2 + (y2_pos - y1_pos)**2)
        total_length += segment_length
    
    return total_length