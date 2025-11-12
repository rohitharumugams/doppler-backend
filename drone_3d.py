# motion_doppler_3d.py
# Fully self-contained module for 3D Doppler calculations for straight, parabolic and Bezier trajectories.
# Returns per-sample frequency ratios and amplitude envelopes suitable for driving an audio Doppler pipeline.

import numpy as np

C = 343.0  # speed of sound (m/s)
DEFAULT_SAMPLE_RATE = 22050


def _compute_num_points(duration, num_points):
    if num_points is None:
        # Conservative frame count: duration * (sr / 100) -> ~220.5 frames per second at 22050 sr
        return max(16, int(duration * DEFAULT_SAMPLE_RATE / 100))
    return max(4, int(num_points))


def _savgol_smooth(arr):
    """Apply a small Savitzky-Golay smoothing if array is large enough and scipy is available."""
    if arr is None or len(arr) <= 10:
        return arr
    try:
        from scipy.signal import savgol_filter
        # choose odd window <= len(arr)
        window_size = min(21, (len(arr) // 20) * 2 + 1)
        if window_size < 3:
            window_size = 3
        return savgol_filter(arr, window_size, 2)
    except Exception:
        return arr


def calculate_straight_line_3d_doppler(speed, h, angle_xy=0, angle_z=0, duration=5, num_points=None):
    """
    Calculate Doppler frequency ratios and amplitude envelope for a 3D straight-line trajectory.

    Args:
        speed (float): vehicle speed in m/s
        h (float): perpendicular distance from observer in meters (in Y direction by convention)
        angle_xy (float): heading angle in XY plane in degrees (0 = +X)
        angle_z (float): elevation angle in degrees (0 = horizontal, positive = upward)
        duration (float): total audio duration in seconds
        num_points (int|None): number of sample points to compute (optional)

    Returns:
        (freq_ratios_list, amplitudes_list)
    """
    num_points = _compute_num_points(duration, num_points)

    # Convert angles to radians
    angle_xy_rad = np.radians(angle_xy)
    angle_z_rad = np.radians(angle_z)

    # Time values centered so closest approach near t=0
    t_vals = np.linspace(-duration / 2.0, duration / 2.0, num_points)

    # Velocity components (constant)
    vx = speed * np.cos(angle_z_rad) * np.cos(angle_xy_rad)
    vy = speed * np.cos(angle_z_rad) * np.sin(angle_xy_rad)
    vz = speed * np.sin(angle_z_rad)

    # Direction perpendicular in XY plane for closest approach offset
    perp_xy = np.array([-np.sin(angle_xy_rad), np.cos(angle_xy_rad)])

    # Closest point coordinates in XY (at t=0)
    closest_x = h * perp_xy[0]
    closest_y = h * perp_xy[1]
    closest_z = 0.0  # by convention; caller can offset with Observer change if needed

    # Positions along the path (linear motion)
    x_arr = closest_x + vx * t_vals
    y_arr = closest_y + vy * t_vals
    z_arr = closest_z + vz * t_vals

    # Distances to observer at origin
    r = np.sqrt(x_arr**2 + y_arr**2 + z_arr**2)
    r = np.maximum(r, 0.001)

    # Radial velocity (dot(v, r_hat))
    vr = (vx * x_arr + vy * y_arr + vz * z_arr) / r

    # Smooth radial velocities if possible
    vr = _savgol_smooth(vr)

    # Clip unsafe extremes
    vr_limited = np.clip(vr, -C * 0.3, C * 0.3)
    freq_ratios = C / (C - vr_limited)

    # Amplitude: inverse distance law with a small floor
    base_amplitude = 1.0 / (r + 1.0)

    # Directional factor: stronger when radial component is large relative to speed
    vel_mag = np.sqrt(vx**2 + vy**2 + vz**2) + 1e-12
    angle_factor = np.abs(vr) / vel_mag
    directional_factor = 1.0 + 0.1 * angle_factor

    amplitudes = base_amplitude * directional_factor

    # Fade-in/out envelope (small portion of signal)
    envelope_length = max(1, min(num_points // 10, int(0.2 * DEFAULT_SAMPLE_RATE)))
    if envelope_length > 1:
        fade_in = np.linspace(0.0, 1.0, envelope_length)
        fade_out = np.linspace(1.0, 0.0, envelope_length)
        amplitudes[:envelope_length] *= fade_in
        amplitudes[-envelope_length:] *= fade_out

    # Logging-like summary (kept non-invasive)
    min_idx = int(np.argmin(r))
    max_radial = float(np.max(np.abs(vr)))
    # Print minimal diagnostic info
    print(f"[straight3d] speed={speed:.2f}m/s duration={duration:.2f}s points={num_points} "
          f"closest_r={r[min_idx]:.2f}m at t={t_vals[min_idx]:.3f}s max_vr={max_radial:.2f}m/s")

    return freq_ratios.tolist(), amplitudes.tolist()


def calculate_parabola_3d_doppler(speed, a, h, z_offset=0.0, duration=5, num_points=None):
    """
    Calculate Doppler frequency ratios and amplitude envelope for a 3D parabolic trajectory.

    The parabola is along X (horizontal) and Z (vertical): z = a * x^2 + z_offset
    The drone moves along X at constant horizontal speed; Y is constant offset 'h'.

    Args:
        speed (float): horizontal speed in m/s (dx/dt)
        a (float): parabola curvature coefficient (z = a*x^2 + z_offset)
        h (float): Y offset (perpendicular distance)
        z_offset (float): vertical offset at x=0 (altitude center)
        duration (float): total audio duration in seconds
        num_points (int|None): number of samples to compute

    Returns:
        (freq_ratios_list, amplitudes_list)
    """
    num_points = _compute_num_points(duration, num_points)
    t_vals = np.linspace(-duration / 2.0, duration / 2.0, num_points)

    # Positions
    x = speed * t_vals
    y = np.full_like(x, h)
    z = a * x**2 + z_offset

    # Velocities
    vx = np.full_like(x, speed)
    vy = np.zeros_like(x)
    vz = 2.0 * a * x * speed  # dz/dt = 2*a*x*(dx/dt)=2*a*x*speed

    # Distances and radial velocities
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.maximum(r, 0.001)
    vr = (vx * x + vy * y + vz * z) / r

    vr = _savgol_smooth(vr)
    vr_limited = np.clip(vr, -C * 0.3, C * 0.3)
    freq_ratios = C / (C - vr_limited)

    base_amplitude = 1.0 / (r + 1.0)
    vel_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    vel_mag = np.maximum(vel_mag, 1e-12)
    angle_factor = np.abs(vr) / vel_mag
    directional_factor = 1.0 + 0.1 * angle_factor
    amplitudes = base_amplitude * directional_factor

    envelope_length = max(1, min(num_points // 10, int(0.2 * DEFAULT_SAMPLE_RATE)))
    if envelope_length > 1:
        fade_in = np.linspace(0.0, 1.0, envelope_length)
        fade_out = np.linspace(1.0, 0.0, envelope_length)
        amplitudes[:envelope_length] *= fade_in
        amplitudes[-envelope_length:] *= fade_out

    min_idx = int(np.argmin(r))
    max_radial = float(np.max(np.abs(vr)))
    print(f"[parabola3d] speed={speed:.2f}m/s duration={duration:.2f}s points={num_points} "
          f"closest_r={r[min_idx]:.2f}m at t={t_vals[min_idx]:.3f}s max_vr={max_radial:.2f}m/s")

    return freq_ratios.tolist(), amplitudes.tolist()


def calculate_bezier_3d_doppler(speed,
                               x0, x1, x2, x3,
                               y0, y1, y2, y3,
                               z0, z1, z2, z3,
                               duration=5,
                               num_points=None):
    """
    Calculate Doppler frequency ratios and amplitude envelope for traversal along a 3D cubic Bezier.

    Args:
        speed (float): traversal speed along the curve in m/s (used to scale parametric derivative)
        x0..x3, y0..y3, z0..z3: control points of the cubic Bezier in meters
        duration (float): total audio duration in seconds
        num_points (int|None): number of sample points

    Returns:
        (freq_ratios_list, amplitudes_list)
    """
    num_points = _compute_num_points(duration, num_points)

    # Compute curve length to decide portion traversed
    curve_length = calculate_bezier_3d_arc_length(x0, x1, x2, x3,
                                                  y0, y1, y2, y3,
                                                  z0, z1, z2, z3)
    total_distance = speed * duration

    # Decide t_start..t_end such that path is centered (closest approach roughly centered)
    if total_distance >= curve_length or curve_length <= 1e-6:
        t_start, t_end = 0.0, 1.0
        actual_distance = curve_length
    else:
        curve_fraction = total_distance / curve_length
        t_center = 0.5
        t_half_span = curve_fraction / 2.0
        t_start = max(0.0, t_center - t_half_span)
        t_end = min(1.0, t_center + t_half_span)
        actual_distance = total_distance

    t_vals = np.linspace(t_start, t_end, num_points)

    # Sample positions and parametric velocities
    positions = np.zeros((len(t_vals), 3))
    velocities_param = np.zeros((len(t_vals), 3))
    for i, t in enumerate(t_vals):
        px = bezier_point(t, x0, x1, x2, x3)
        py = bezier_point(t, y0, y1, y2, y3)
        pz = bezier_point(t, z0, z1, z2, z3)
        positions[i, :] = [px, py, pz]

        vx_p = bezier_velocity(t, x0, x1, x2, x3)
        vy_p = bezier_velocity(t, y0, y1, y2, y3)
        vz_p = bezier_velocity(t, z0, z1, z2, z3)
        velocities_param[i, :] = [vx_p, vy_p, vz_p]

    # Convert parametric derivative to metric velocity by scaling each vector so its magnitude = desired speed
    param_speeds = np.linalg.norm(velocities_param, axis=1)
    velocities = np.zeros_like(velocities_param)
    for i in range(len(param_speeds)):
        if param_speeds[i] > 1e-9:
            scale_factor = speed / param_speeds[i]
            velocities[i, :] = velocities_param[i, :] * scale_factor
        else:
            # If parametric derivative nearly zero, set small forward velocity along chord direction
            if i < len(positions) - 1:
                chord = positions[min(i + 1, len(positions) - 1)] - positions[max(i - 1, 0)]
                chord_norm = np.linalg.norm(chord)
                velocities[i, :] = (chord / (chord_norm + 1e-9)) * speed
            else:
                velocities[i, :] = np.array([speed, 0.0, 0.0])

    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    z_coords = positions[:, 2]
    distances = np.sqrt(x_coords**2 + y_coords**2 + z_coords**2)
    distances = np.maximum(distances, 0.001)

    vx_arr = velocities[:, 0]
    vy_arr = velocities[:, 1]
    vz_arr = velocities[:, 2]

    radial_velocities = (vx_arr * x_coords + vy_arr * y_coords + vz_arr * z_coords) / distances
    radial_velocities = _savgol_smooth(radial_velocities)
    vr_limited = np.clip(radial_velocities, -C * 0.3, C * 0.3)
    freq_ratios = C / (C - vr_limited)

    base_amplitude = 1.0 / (distances + 1.0)
    vel_mags = np.sqrt(vx_arr**2 + vy_arr**2 + vz_arr**2)
    vel_mags = np.maximum(vel_mags, 1e-12)
    angle_factors = np.abs(radial_velocities) / vel_mags
    directional_factors = 1.0 + 0.1 * angle_factors
    amplitudes = base_amplitude * directional_factors

    envelope_length = max(1, min(num_points // 10, int(0.2 * DEFAULT_SAMPLE_RATE)))
    if envelope_length > 1:
        fade_in = np.linspace(0.0, 1.0, envelope_length)
        fade_out = np.linspace(1.0, 0.0, envelope_length)
        amplitudes[:envelope_length] *= fade_in
        amplitudes[-envelope_length:] *= fade_out

    min_idx = int(np.argmin(distances))
    max_radial = float(np.max(np.abs(radial_velocities)))
    print(f"[bezier3d] speed={speed:.2f}m/s duration={duration:.2f}s points={num_points} "
          f"curve_len={curve_length:.2f}m traveled={actual_distance:.2f}m "
          f"closest_r={distances[min_idx]:.2f}m at t={t_vals[min_idx]:.3f}s max_vr={max_radial:.2f}m/s")

    return freq_ratios.tolist(), amplitudes.tolist()


def bezier_point(t, p0, p1, p2, p3):
    """Cubic Bezier point at parameter t (scalar inputs)."""
    omt = 1.0 - t
    return (omt**3) * p0 + 3.0 * (omt**2) * t * p1 + 3.0 * omt * (t**2) * p2 + (t**3) * p3


def bezier_velocity(t, p0, p1, p2, p3):
    """Derivative of cubic Bezier at t (parametric velocity)."""
    return 3.0 * (1.0 - t)**2 * (p1 - p0) + 6.0 * (1.0 - t) * t * (p2 - p1) + 3.0 * t**2 * (p3 - p2)


def calculate_bezier_3d_arc_length(x0, x1, x2, x3,
                                   y0, y1, y2, y3,
                                   z0, z1, z2, z3,
                                   num_segments=1000):
    """
    Approximate arc length of a 3D cubic Bezier curve using piecewise linear sampling.
    """
    if num_segments < 4:
        num_segments = 4
    t_vals = np.linspace(0.0, 1.0, num_segments + 1)
    pts_prev = np.array([bezier_point(t_vals[0], x0, x1, x2, x3),
                         bezier_point(t_vals[0], y0, y1, y2, y3),
                         bezier_point(t_vals[0], z0, z1, z2, z3)])
    total_length = 0.0
    for i in range(1, len(t_vals)):
        t = t_vals[i]
        pts_curr = np.array([bezier_point(t, x0, x1, x2, x3),
                             bezier_point(t, y0, y1, y2, y3),
                             bezier_point(t, z0, z1, z2, z3)])
        total_length += np.linalg.norm(pts_curr - pts_prev)
        pts_prev = pts_curr
    return float(total_length)


# If module is executed directly, run a small sanity check
if __name__ == "__main__":
    # Quick smoke test for each function
    fr_s, amp_s = calculate_straight_line_3d_doppler(speed=20, h=10, angle_xy=0, angle_z=10, duration=3)
    fr_p, amp_p = calculate_parabola_3d_doppler(speed=15, a=0.02, h=12, z_offset=5, duration=4)
    fr_b, amp_b = calculate_bezier_3d_doppler(
        speed=12,
        x0=-30, x1=-10, x2=10, x3=30,
        y0=20, y1=20, y2=20, y3=20,
        z0=10, z1=15, z2=15, z3=10,
        duration=4
    )
    print("Straight sample ratios (first 5):", fr_s[:5])
    print("Parabola sample ratios (first 5):", fr_p[:5])
    print("Bezier sample ratios (first 5):", fr_b[:5])
