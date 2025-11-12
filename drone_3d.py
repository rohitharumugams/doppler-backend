import numpy as np

def calculate_straight_line_3d_doppler(speed, h, angle_xy=0, angle_z=0, duration=5, num_points=None):
    """
    Calculate Doppler effect for 3D straight line motion
    
    Args:
        speed: vehicle speed in m/s
        h: perpendicular distance from observer in m
        angle_xy: angle in XY plane in degrees (0 = along +X axis)
        angle_z: elevation angle in degrees (0 = horizontal, positive = upward)
        duration: total audio duration in seconds
        num_points: number of calculation points
    
    Returns:
        tuple: (freq_ratios, amplitudes)
    """
    c = 343.0  # Speed of sound in m/s
    
    if num_points is None:
        num_points = int(duration * 22050 / 100)
    
    # Convert angles to radians
    angle_xy_rad = np.radians(angle_xy)
    angle_z_rad = np.radians(angle_z)
    
    # Calculate total distance traveled
    total_distance = speed * duration
    
    # Time range centered so closest approach is in the middle
    t_vals = np.linspace(-duration / 2, duration / 2, num_points)
    
    print(f"3D Straight line motion:")
    print(f"  Speed: {speed} m/s")
    print(f"  Duration: {duration} s")
    print(f"  Total distance: {total_distance:.1f} m")
    print(f"  Perpendicular distance: {h} m")
    print(f"  XY Angle: {angle_xy}°")
    print(f"  Z Angle (elevation): {angle_z}°")
    print(f"  Audio samples: {num_points}")
    
    # Velocity components in 3D
    # First project onto XY plane, then add Z component
    vx = speed * np.cos(angle_z_rad) * np.cos(angle_xy_rad)
    vy = speed * np.cos(angle_z_rad) * np.sin(angle_xy_rad)
    vz = speed * np.sin(angle_z_rad)
    
    print(f"  Velocity components: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f} m/s")
    
    # Calculate closest point on path to observer (at origin)
    # For a straight line, the perpendicular distance h is maintained
    # The closest point is at t=0 (center of time range)
    
    # Direction vector (normalized)
    dir_xy = np.array([np.cos(angle_xy_rad), np.sin(angle_xy_rad)])
    
    # Perpendicular vector in XY plane
    perp_xy = np.array([-np.sin(angle_xy_rad), np.cos(angle_xy_rad)])
    
    # Closest point in 3D
    closest_x = h * perp_xy[0]
    closest_y = h * perp_xy[1]
    closest_z = 0  # At t=0, we set z=0 for simplicity
    
    # Position along path at each time
    x_arr = closest_x + vx * t_vals
    y_arr = closest_y + vy * t_vals
    z_arr = closest_z + vz * t_vals
    
    # 3D distance from observer (at origin)
    r = np.sqrt(x_arr**2 + y_arr**2 + z_arr**2)
    
    # Prevent division by zero
    r = np.maximum(r, 0.1)
    
    # Radial velocity (positive = approaching observer)
    # vr = (v⃗ · r⃗) / |r⃗|
    vr = (vx * x_arr + vy * y_arr + vz * z_arr) / r
    
    # Apply velocity smoothing
    if len(vr) > 10:
        from scipy.signal import savgol_filter
        window_size = min(21, len(vr) // 20 * 2 + 1)
        if window_size >= 3:
            try:
                vr = savgol_filter(vr, window_size, 2)
            except:
                pass
    
    # Doppler frequency ratio: f'/f = c/(c - vr)
    vr_limited = np.clip(vr, -c*0.3, c*0.3)
    freq_ratios = c / (c - vr_limited)
    
    # Amplitude calculation (inverse distance law)
    base_amplitude = 1.0 / (r + 1.0)
    
    # Directional effects
    velocity_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
    angle_factor = np.abs(vr) / (velocity_magnitude + 1e-6)
    directional_factor = 1.0 + 0.1 * angle_factor
    
    amplitudes = base_amplitude * directional_factor
    
    # Apply envelope
    envelope_length = min(num_points // 10, int(0.2 * 22050))
    if envelope_length > 0:
        fade_in = np.linspace(0, 1, envelope_length)
        fade_out = np.linspace(1, 0, envelope_length)
        amplitudes[:envelope_length] *= fade_in
        amplitudes[-envelope_length:] *= fade_out
    
    # Find closest approach
    min_distance_idx = np.argmin(r)
    min_distance = r[min_distance_idx]
    closest_t = t_vals[min_distance_idx]
    max_radial_vel = np.max(np.abs(vr))
    
    print(f"  Closest approach: {min_distance:.2f} m at t={closest_t:.3f} s")
    print(f"  Closest point: ({x_arr[min_distance_idx]:.2f}, {y_arr[min_distance_idx]:.2f}, {z_arr[min_distance_idx]:.2f}) m")
    print(f"  Max radial velocity: {max_radial_vel:.2f} m/s ({max_radial_vel/c*100:.1f}% of c)")
    print(f"  Frequency ratio range: {np.min(freq_ratios):.3f} to {np.max(freq_ratios):.3f}")
    print(f"  Start position: ({x_arr[0]:.1f}, {y_arr[0]:.1f}, {z_arr[0]:.1f}) m")
    print(f"  End position: ({x_arr[-1]:.1f}, {y_arr[-1]:.1f}, {z_arr[-1]:.1f}) m")
    
    return freq_ratios.tolist(), amplitudes.tolist()


def calculate_parabola_3d_doppler(speed, a, h, z_offset=0, duration=5, num_points=None):
    """
    Calculate Doppler effect for 3D parabolic motion
    
    The parabola is in the XZ plane: z = a*x² + z_offset
    The drone moves along this path at constant horizontal speed
    
    Args:
        speed: horizontal speed in m/s
        a: parabola coefficient (curvature)
        h: perpendicular distance in Y direction in meters
        z_offset: vertical offset (altitude at x=0) in meters
        duration: total audio duration in seconds
        num_points: number of calculation points
    
    Returns:
        tuple: (freq_ratios, amplitudes)
    """
    c = 343.0  # Speed of sound in m/s
    
    if num_points is None:
        num_points = int(duration * 22050 / 100)
    
    # Calculate total distance traveled
    total_distance = speed * duration
    
    # Time array centered around t=0
    t_vals = np.linspace(-duration/2, duration/2, num_points)
    
    print(f"3D Parabolic motion:")
    print(f"  Speed: {speed} m/s")
    print(f"  Duration: {duration} s")
    print(f"  Total distance: {total_distance:.1f} m")
    print(f"  Curvature (a): {a}")
    print(f"  Y-distance (h): {h} m")
    print(f"  Z-offset (altitude at center): {z_offset} m")
    print(f"  Audio samples: {num_points}")
    
    # Position: parabola in XZ plane, constant Y
    x = speed * t_vals
    y = np.full_like(x, h)  # Constant Y offset
    z = a * x**2 + z_offset  # Parabolic Z
    
    # Velocity components
    vx = np.full_like(x, speed)  # Constant horizontal velocity
    vy = np.zeros_like(x)  # No Y movement
    vz = 2 * a * x * speed  # dz/dt = 2a * x * dx/dt
    
    # 3D distance from observer
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.maximum(r, 0.1)
    
    # Radial velocity
    vr = (vx * x + vy * y + vz * z) / r
    
    # Apply smoothing
    if len(vr) > 10:
        from scipy.signal import savgol_filter
        window_size = min(21, len(vr) // 20 * 2 + 1)
        if window_size >= 3:
            try:
                vr = savgol_filter(vr, window_size, 2)
            except:
                pass
    
    # Doppler frequency ratio
    vr_limited = np.clip(vr, -c*0.3, c*0.3)
    freq_ratios = c / (c - vr_limited)
    
    # Amplitude calculation
    base_amplitude = 1.0 / (r + 1.0)
    
    velocity_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
    angle_factor = np.abs(vr) / (velocity_magnitude + 1e-6)
    directional_factor = 1.0 + 0.1 * angle_factor
    
    amplitudes = base_amplitude * directional_factor
    
    # Apply envelope
    envelope_length = min(num_points // 10, int(0.2 * 22050))
    if envelope_length > 0:
        fade_in = np.linspace(0, 1, envelope_length)
        fade_out = np.linspace(1, 0, envelope_length)
        amplitudes[:envelope_length] *= fade_in
        amplitudes[-envelope_length:] *= fade_out
    
    # Find closest approach
    min_distance_idx = np.argmin(r)
    min_distance = r[min_distance_idx]
    closest_t = t_vals[min_distance_idx]
    max_radial_vel = np.max(np.abs(vr))
    
    print(f"  Closest approach: {min_distance:.2f} m at t={closest_t:.3f} s")
    print(f"  Closest point: ({x[min_distance_idx]:.2f}, {y[min_distance_idx]:.2f}, {z[min_distance_idx]:.2f}) m")
    print(f"  Max radial velocity: {max_radial_vel:.2f} m/s ({max_radial_vel/c*100:.1f}% of c)")
    print(f"  Frequency ratio range: {np.min(freq_ratios):.3f} to {np.max(freq_ratios):.3f}")
    print(f"  Start position: ({x[0]:.1f}, {y[0]:.1f}, {z[0]:.1f}) m")
    print(f"  End position: ({x[-1]:.1f}, {y[-1]:.1f}, {z[-1]:.1f}) m")
    
    return freq_ratios.tolist(), amplitudes.tolist()


def calculate_bezier_3d_doppler(speed, x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3, duration=5, num_points=None):
    """
    Calculate Doppler effect for 3D Bezier curve motion
    
    Args:
        speed: vehicle speed in m/s (controls traversal rate)
        x0, y0, z0: Start point coordinates (m)
        x1, y1, z1: First control point coordinates (m)
        x2, y2, z2: Second control point coordinates (m)
        x3, y3, z3: End point coordinates (m)
        duration: total audio duration in seconds
        num_points: number of calculation points
    
    Returns:
        tuple: (freq_ratios, amplitudes)
    """
    c = 343.0  # Speed of sound in m/s
    
    if num_points is None:
        num_points = int(duration * 22050 / 100)
    
    # Calculate the total arc length of the 3D Bezier curve
    curve_length = calculate_bezier_3d_arc_length(x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3)
    
    # Calculate how much of the curve we traverse
    total_distance = speed * duration
    
    # Determine what portion of the curve we actually travel
    if total_distance >= curve_length:
        t_start = 0.0
        t_end = 1.0
        actual_distance = curve_length
        print(f"  NOTE: Speed allows traveling entire curve ({curve_length:.1f}m)")
    else:
        curve_fraction = total_distance / curve_length
        t_center = 0.5
        t_half_span = curve_fraction / 2
        t_start = max(0.0, t_center - t_half_span)
        t_end = min(1.0, t_center + t_half_span)
        actual_distance = total_distance
        print(f"  Traveling {curve_fraction*100:.1f}% of curve (center portion)")
    
    t_vals = np.linspace(t_start, t_end, num_points)
    
    print(f"3D Bezier curve motion:")
    print(f"  Speed: {speed} m/s")
    print(f"  Duration: {duration} s")
    print(f"  Total curve length: {curve_length:.1f} m")
    print(f"  Distance at this speed: {total_distance:.1f} m")
    print(f"  Actual distance traveled: {actual_distance:.1f} m")
    print(f"  P0: ({x0:.1f}, {y0:.1f}, {z0:.1f}) m")
    print(f"  P1: ({x1:.1f}, {y1:.1f}, {z1:.1f}) m")
    print(f"  P2: ({x2:.1f}, {y2:.1f}, {z2:.1f}) m")
    print(f"  P3: ({x3:.1f}, {y3:.1f}, {z3:.1f}) m")
    print(f"  t range: {t_start:.3f} to {t_end:.3f}")
    print(f"  Audio samples: {num_points}")
    
    # Calculate position and velocity along 3D Bezier curve
    positions = []
    velocities = []
    
    for t in t_vals:
        # 3D Bezier position
        pos_x = bezier_point(t, x0, x1, x2, x3)
        pos_y = bezier_point(t, y0, y1, y2, y3)
        pos_z = bezier_point(t, z0, z1, z2, z3)
        positions.append((pos_x, pos_y, pos_z))
        
        # 3D Bezier velocity (parametric)
        vel_x_param = bezier_velocity(t, x0, x1, x2, x3)
        vel_y_param = bezier_velocity(t, y0, y1, y2, y3)
        vel_z_param = bezier_velocity(t, z0, z1, z2, z3)
        
        # Calculate parametric speed
        param_speed = np.sqrt(vel_x_param**2 + vel_y_param**2 + vel_z_param**2)
        
        # Scale to match actual speed
        if param_speed > 1e-6:
            scale_factor = speed / param_speed
            vel_x = vel_x_param * scale_factor
            vel_y = vel_y_param * scale_factor
            vel_z = vel_z_param * scale_factor
        else:
            vel_x = speed
            vel_y = 0
            vel_z = 0
        
        velocities.append((vel_x, vel_y, vel_z))
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    # 3D distances from observer
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    z_coords = positions[:, 2]
    distances = np.sqrt(x_coords**2 + y_coords**2 + z_coords**2)
    distances = np.maximum(distances, 0.1)
    
    # Radial velocity
    vx_arr = velocities[:, 0]
    vy_arr = velocities[:, 1]
    vz_arr = velocities[:, 2]
    radial_velocities = (vx_arr * x_coords + vy_arr * y_coords + vz_arr * z_coords) / distances
    
    # Apply smoothing
    if len(radial_velocities) > 10:
        from scipy.signal import savgol_filter
        window_size = min(21, len(radial_velocities) // 20 * 2 + 1)
        if window_size >= 3:
            try:
                radial_velocities = savgol_filter(radial_velocities, window_size, 2)
            except:
                pass
    
    # Doppler frequency ratio
    vr_limited = np.clip(radial_velocities, -c*0.3, c*0.3)
    freq_ratios = c / (c - vr_limited)
    
    # Amplitude calculation
    base_amplitude = 1.0 / (distances + 1.0)
    
    velocity_magnitudes = np.sqrt(vx_arr**2 + vy_arr**2 + vz_arr**2)
    velocity_magnitudes = np.maximum(velocity_magnitudes, 1e-6)
    angle_factors = np.abs(radial_velocities) / velocity_magnitudes
    directional_factors = 1.0 + 0.1 * angle_factors
    
    amplitudes = base_amplitude * directional_factors
    
    # Apply envelope
    envelope_length = min(num_points // 10, int(0.2 * 22050))
    if envelope_length > 0:
        fade_in = np.linspace(0, 1, envelope_length)
        fade_out = np.linspace(1, 0, envelope_length)
        amplitudes[:envelope_length] *= fade_in
        amplitudes[-envelope_length:] *= fade_out
    
    # Find closest approach
    min_distance_idx = np.argmin(distances)
    min_distance = distances[min_distance_idx]
    closest_t = t_vals[min_distance_idx]
    closest_pos = positions[min_distance_idx]
    max_radial_vel = np.max(np.abs(radial_velocities))
    
    print(f"  Closest approach: {min_distance:.2f} m at t={closest_t:.3f}")
    print(f"  Closest point: ({closest_pos[0]:.2f}, {closest_pos[1]:.2f}, {closest_pos[2]:.2f}) m")
    print(f"  Max radial velocity: {max_radial_vel:.2f} m/s ({max_radial_vel/c*100:.1f}% of c)")
    print(f"  Frequency ratio range: {np.min(freq_ratios):.3f} to {np.max(freq_ratios):.3f}")
    print(f"  Amplitude range: {np.min(amplitudes):.4f} to {np.max(amplitudes):.4f}")
    
    avg_speed = actual_distance / duration
    max_speed = np.max(velocity_magnitudes)
    
    print(f"  Curve length: {curve_length:.1f} m")
    print(f"  Average speed: {avg_speed:.1f} m/s")
    print(f"  Maximum speed: {max_speed:.1f} m/s")
    print(f"  Start position: ({x0:.1f}, {y0:.1f}, {z0:.1f}) m")
    print(f"  End position: ({x3:.1f}, {y3:.1f}, {z3:.1f}) m")
    
    return freq_ratios.tolist(), amplitudes.tolist()


def bezier_point(t, p0, p1, p2, p3):
    """Calculate a point on a cubic Bezier curve at parameter t"""
    return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def bezier_velocity(t, p0, p1, p2, p3):
    """Calculate velocity (derivative) of a cubic Bezier curve at parameter t"""
    return 3 * (1 - t)**2 * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * t**2 * (p3 - p2)


def calculate_bezier_3d_arc_length(x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3, num_segments=1000):
    """
    Calculate approximate arc length of 3D Bezier curve using numerical integration
    
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
        z1_pos = bezier_point(t1, z0, z1, z2, z3)
        
        x2_pos = bezier_point(t2, x0, x1, x2, x3)
        y2_pos = bezier_point(t2, y0, y1, y2, y3)
        z2_pos = bezier_point(t2, z0, z1, z2, z3)
        
        # Add 3D segment length
        segment_length = np.sqrt((x2_pos - x1_pos)**2 + (y2_pos - y1_pos)**2 + (z2_pos - z1_pos)**2)
        total_length += segment_length
    
    return total_length