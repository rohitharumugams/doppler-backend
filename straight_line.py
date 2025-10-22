import numpy as np

def calculate_straight_line_doppler(speed, h, angle=0, duration=5, num_points=200):
    """
    Calculate Doppler effect for straight line motion based on speed and duration
    
    Args:
        speed: vehicle speed in m/s
        h: perpendicular distance in m
        angle: angle of motion in degrees (0 = horizontal)
        duration: total audio duration in seconds
        num_points: number of calculation points
    
    Returns:
        tuple: (freq_ratios, amplitudes)
    """
    c = 343  # Speed of sound in m/s
    
    # Convert angle to radians and calculate slope
    angle_rad = np.radians(angle)
    m = np.tan(angle_rad)
    
    # Calculate total distance traveled during audio duration
    total_distance = speed * duration
    
    # Time range centered so closest approach is in the middle
    t_vals = np.linspace(-duration / 2, duration / 2, num_points)
    
    freq_ratios = []
    amplitudes = []
    
    print(f"Straight line motion:")
    print(f"  Speed: {speed} m/s")
    print(f"  Duration: {duration} s")
    print(f"  Total distance: {total_distance:.1f} m")
    print(f"  Perpendicular distance: {h} m")
    print(f"  Angle: {angle}°")
    
    # Calculate c0 from perpendicular distance
    c0 = h * np.sqrt(1 + m**2)
    
    # Calculate distance range
    half_distance = total_distance / 2
    start_distance = -half_distance
    end_distance = half_distance
    
    print(f"  Path: {start_distance:.1f} m to {end_distance:.1f} m")
    
    for t in t_vals:
        if abs(m) < 1e-6:  # Essentially horizontal (angle ≈ 0)
            # Simple horizontal case
            x = speed * t
            distance = np.sqrt(x**2 + h**2)
            vr = (speed * x) / distance
            # FIXED: Changed from c / (c + vr) to c / (c - vr) for consistency
            freq_ratio = c / (c - vr)
            R_t = distance
            
        elif m >= 0:  # Positive slope
            # Position along the line
            x_pos = speed * t * np.cos(angle_rad)
            y_pos = h + x_pos * m
            
            # Distance from observer
            R_t = np.sqrt(x_pos**2 + y_pos**2)
            
            # Velocity components
            vx = speed * np.cos(angle_rad)
            vy = speed * np.sin(angle_rad)
            
            # Radial velocity
            if R_t > 0:
                vr = (vx * x_pos + vy * y_pos) / R_t
            else:
                vr = 0
                
            freq_ratio = c / (c - vr)
            
        else:  # Negative slope (m < 0)
            # Position along the line
            x_pos = speed * t * np.cos(angle_rad)
            y_pos = h + x_pos * m
            
            # Distance from observer
            R_t = np.sqrt(x_pos**2 + y_pos**2)
            
            # Velocity components
            vx = speed * np.cos(angle_rad)
            vy = speed * np.sin(angle_rad)
            
            # Radial velocity
            if R_t > 0:
                vr = (vx * x_pos + vy * y_pos) / R_t
            else:
                vr = 0
                
            freq_ratio = c / (c - vr)
        
        freq_ratios.append(freq_ratio)
        
        # FIXED: More physical amplitude calculation (inverse distance law)
        amp = 1.0 / (R_t + 1.0)
        amplitudes.append(amp)
    
    print(f"  Frequency ratios: {min(freq_ratios):.3f} to {max(freq_ratios):.3f}")
    print(f"  Distance range: {min([np.sqrt((speed*t)**2 + h**2) for t in t_vals]):.1f} to {max([np.sqrt((speed*t)**2 + h**2) for t in t_vals]):.1f} m")
    
    return freq_ratios, amplitudes