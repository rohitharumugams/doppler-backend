# realistic_simulation.py - Merged realistic vehicle simulation module
# FIXED: Improved Bezier manual profile calculations and optional velocity smoothing
# ADDED: 'drone' vehicle profile

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

# Vehicle-specific profiles with realistic parameters
VEHICLE_PROFILES = {
    'car': {
        'name': 'Passenger Car',
        'max_acceleration': (2.0, 4.0),  # m/s² (min, max)
        'max_deceleration': (3.0, 8.0),  # m/s² (min, max)
        'acceleration_frequency': (0.1, 0.3),  # events per second (min, max)
        'acceleration_strength': (10, 30),  # percentage of max acceleration (min, max)
        'gear_change_frequency': (0.02, 0.08),  # gear changes per second (min, max)
        'engine_roughness': (5, 20),  # engine roughness percentage (min, max)
        'typical_speeds': (5, 60),  # typical speed range (min, max) m/s
        'path_deviation': (0.1, 0.5),  # lateral deviation in meters (min, max)
        'mass': 1500,  # kg
        'drag_coefficient': 0.3
    },
    'train': {
        'name': 'Heavy Train',
        'max_acceleration': (0.5, 1.5),  # Slower acceleration due to mass
        'max_deceleration': (0.8, 2.5),
        'acceleration_frequency': (0.05, 0.15),  # Less frequent speed changes
        'acceleration_strength': (5, 20),  # More gradual changes
        'gear_change_frequency': (0.01, 0.03),  # Diesel engines
        'engine_roughness': (10, 35),  # Diesel engine characteristics
        'typical_speeds': (10, 120),  # Higher max speeds
        'path_deviation': (0.05, 0.2),  # Trains stay on rails
        'mass': 500000,  # kg (much heavier)
        'drag_coefficient': 0.6
    },
    'flight': {
        'name': 'Aircraft',
        'max_acceleration': (1.0, 3.0),  # Turbine characteristics
        'max_deceleration': (1.5, 4.0),
        'acceleration_frequency': (0.02, 0.08),  # Smoother flight
        'acceleration_strength': (5, 15),  # Gradual thrust changes
        'gear_change_frequency': (0.005, 0.02),  # Turbine adjustments
        'engine_roughness': (2, 10),  # Smoother turbine operation
        'typical_speeds': (50, 300),  # Flight speeds
        'path_deviation': (0.5, 2.0),  # Air turbulence effects
        'mass': 75000,  # kg
        'drag_coefficient': 0.02  # More aerodynamic
    },
    'drone': {
        'name': 'Multirotor Drone',
        'max_acceleration': (2.0, 8.0),  # m/s² (min, max) - agile vertical/horizontal accel
        'max_deceleration': (2.0, 8.0),
        'acceleration_frequency': (0.2, 0.6),  # more frequent micro-maneuvers
        'acceleration_strength': (20, 80),  # percentage of max acceleration (min, max)
        'gear_change_frequency': (0.05, 0.2),  # throttle adjustments
        'engine_roughness': (5, 25),  # rotor-induced vibrations
        'typical_speeds': (1, 25),  # typical small UAV speeds (m/s)
        'path_deviation': (0.2, 5.0),  # allows larger lateral/vertical deviation
        'mass': 5.0,  # kg (small/medium drone)
        'drag_coefficient': 0.5  # relatively draggy at low Reynolds
        # Note: drones frequently operate in 3D; path generators can be extended to include z
    }
}

# Example preset profiles for quick selection
PRESET_MANUAL_PROFILES = {
    "city_traffic": {
        "name": "City Traffic (Stop & Go)",
        "description": "Typical city driving with stops and acceleration",
        "time_values": "0,2,4,6,8,10",
        "speed_values": "0,25,10,30,5,20",
        "duration": 10
    },
    "highway_cruise": {
        "name": "Highway Cruise Control",
        "description": "Steady highway driving with minor variations",
        "time_values": "0,1,5,9,10",
        "speed_values": "25,30,28,32,30",
        "duration": 10
    },
    "acceleration_test": {
        "name": "Acceleration Test",
        "description": "Gradual acceleration from stop to high speed",
        "time_values": "0,2,5,8,10",
        "speed_values": "5,15,35,50,55",
        "duration": 10
    },
    "mountain_driving": {
        "name": "Mountain Driving",
        "description": "Uphill/downhill with varying speeds",
        "time_values": "0,3,6,9,12,15",
        "speed_values": "20,15,10,25,35,20",
        "duration": 15
    },
    "racing_circuit": {
        "name": "Racing Circuit",
        "description": "High-speed racing with braking zones",
        "time_values": "0,1,3,5,7,8",
        "speed_values": "40,60,30,70,35,50",
        "duration": 8
    }
}


def parse_manual_acceleration_profile(time_values_str, speed_values_str, duration):
    """
    Parse manual acceleration profile from user input strings

    Args:
        time_values_str: Comma-separated time values (e.g., "0,1,2,3,4,5")
        speed_values_str: Comma-separated speed values (e.g., "20,25,15,30,10,25")
        duration: Total duration in seconds

    Returns:
        tuple: (time_array, speed_array, profile_info)
    """
    try:
        # Parse input strings
        time_points = [float(x.strip()) for x in time_values_str.split(',') if x.strip()]
        speed_points = [float(x.strip()) for x in speed_values_str.split(',') if x.strip()]

        if len(time_points) != len(speed_points):
            raise ValueError("Number of time points must match number of speed points")

        if len(time_points) < 2:
            raise ValueError("Need at least 2 time/speed points")

        # Ensure time points are sorted and within duration
        time_points = np.array(time_points)
        speed_points = np.array(speed_points)

        # Sort by time
        sort_indices = np.argsort(time_points)
        time_points = time_points[sort_indices]
        speed_points = speed_points[sort_indices]

        # Ensure first point is at t=0 and last point is at duration
        if time_points[0] > 0:
            time_points = np.insert(time_points, 0, 0)
            speed_points = np.insert(speed_points, 0, speed_points[0])

        if time_points[-1] < duration:
            time_points = np.append(time_points, duration)
            speed_points = np.append(speed_points, speed_points[-1])

        # Clip time points to duration
        time_points = np.clip(time_points, 0, duration)

        # Create high-resolution time array
        dt = 0.01
        time_array = np.arange(0, duration + dt, dt)

        # Interpolate speed profile
        speed_interp = interp1d(time_points, speed_points, kind='cubic',
                               bounds_error=False, fill_value='extrapolate')
        speed_array = speed_interp(time_array)

        # Ensure positive speeds
        speed_array = np.maximum(speed_array, 0.1)

        profile_info = {
            "variation_type": "manual_profile",
            "input_points": len(time_points),
            "time_points": time_points.tolist(),
            "speed_points": speed_points.tolist(),
            "min_speed": np.min(speed_array),
            "max_speed": np.max(speed_array),
            "avg_speed": np.mean(speed_array),
            "speed_std": np.std(speed_array)
        }

        print(f"Manual acceleration profile:")
        print(f"  Input points: {len(time_points)}")
        print(f"  Time range: {time_points[0]:.1f} to {time_points[-1]:.1f} s")
        print(f"  Speed range: {profile_info['min_speed']:.1f} to {profile_info['max_speed']:.1f} m/s")
        print(f"  Speed variation: ±{profile_info['speed_std']:.1f} m/s")

        return time_array, speed_array, profile_info

    except Exception as e:
        print(f"Error parsing manual profile: {e}")
        raise ValueError(f"Invalid manual profile input: {e}")


def generate_custom_speed_profile(base_speed, duration, vehicle_type='car', custom_params=None):
    """
    Generate speed profile with custom acceleration parameters

    Args:
        base_speed: Target average speed in m/s
        duration: Duration in seconds
        vehicle_type: 'car', 'train', 'flight' or 'drone'
        custom_params: Custom parameters to override defaults

    Returns:
        tuple: (time_array, speed_array, profile_info)
    """
    print(f"Generating custom speed profile:")
    print(f"  Vehicle: {vehicle_type}")
    print(f"  Base speed: {base_speed} m/s")
    print(f"  Duration: {duration} s")

    # Get vehicle profile
    profile = VEHICLE_PROFILES.get(vehicle_type, VEHICLE_PROFILES['car']).copy()

    # Create high-resolution time array
    dt = 0.01  # 10ms resolution
    time_array = np.arange(0, duration + dt, dt)

    # Apply custom parameters if provided
    if custom_params:
        print(f"  Applying custom parameters: {custom_params}")
        for key, value in custom_params.items():
            if key in profile:
                if isinstance(profile[key], tuple):
                    # Replace tuple with custom value as both min and max
                    profile[key] = (value, value)
                else:
                    profile[key] = value

    # Extract parameters (use max values for custom mode)
    max_accel = profile['max_acceleration'][1]
    max_decel = profile['max_deceleration'][1]
    accel_freq = profile['acceleration_frequency'][1]
    accel_strength = profile['acceleration_strength'][1] / 100.0
    gear_freq = profile['gear_change_frequency'][1]
    engine_roughness = profile['engine_roughness'][1] / 100.0

    print(f"  Max acceleration: {max_accel:.2f} m/s²")
    print(f"  Max deceleration: {max_decel:.2f} m/s²")
    print(f"  Acceleration frequency: {accel_freq:.3f} events/s")
    print(f"  Engine roughness: {engine_roughness*100:.1f}%")

    # Initialize speed array with base speed
    speed_array = np.full(len(time_array), base_speed)

    # 1. Add acceleration/deceleration events
    num_accel_events = int(accel_freq * duration)
    if num_accel_events > 0:
        event_times = np.random.uniform(0, duration, num_accel_events)

        for event_time in event_times:
            event_idx = int(event_time / dt)
            if event_idx < len(time_array):
                # Random acceleration or deceleration
                is_acceleration = np.random.choice([True, False])
                if is_acceleration:
                    accel_magnitude = np.random.uniform(0.2 * max_accel, max_accel)
                else:
                    accel_magnitude = -np.random.uniform(0.2 * max_decel, max_decel)

                # Event duration
                event_duration = np.random.uniform(0.5, 3.0)  # 0.5 to 3 seconds
                event_samples = int(event_duration / dt)

                # Apply acceleration with smooth ramp up/down
                for i in range(event_samples):
                    if event_idx + i < len(time_array):
                        # Smooth acceleration profile (bell curve)
                        progress = i / event_samples
                        smooth_factor = np.sin(progress * np.pi)  # 0 to 1 to 0
                        acceleration = accel_magnitude * smooth_factor * accel_strength

                        # Integrate acceleration to get speed change
                        speed_array[event_idx + i] += acceleration * dt

    # 2. Add gear changes (sudden small speed adjustments)
    num_gear_events = int(gear_freq * duration)
    if num_gear_events > 0:
        gear_times = np.random.uniform(0, duration, num_gear_events)

        for gear_time in gear_times:
            gear_idx = int(gear_time / dt)
            if gear_idx < len(time_array):
                # Small speed adjustment (±2-5% of current speed)
                speed_change_percent = np.random.uniform(-0.05, 0.05)
                speed_change = speed_array[gear_idx] * speed_change_percent

                # Apply over short duration (0.1-0.3 seconds)
                gear_duration = np.random.uniform(0.1, 0.3)
                gear_samples = int(gear_duration / dt)

                for i in range(gear_samples):
                    if gear_idx + i < len(time_array):
                        speed_array[gear_idx + i] += speed_change

    # 3. Add engine roughness (high-frequency variations)
    if engine_roughness > 0:
        # Generate noise with vehicle-specific characteristics
        noise_freq = np.random.uniform(10, 50)  # Hz
        noise_amplitude = base_speed * engine_roughness * 0.1

        # Create multiple noise components for realistic engine behavior
        engine_noise = np.zeros(len(time_array))
        for freq_mult in [1.0, 1.5, 2.0, 2.7]:  # Harmonic components
            frequency = noise_freq * freq_mult
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = noise_amplitude / freq_mult  # Higher frequencies have lower amplitude

            engine_noise += amplitude * np.sin(2 * np.pi * frequency * time_array + phase)

        # Add engine noise
        speed_array += engine_noise

    # 4. Apply vehicle-specific speed constraints
    min_speed = max(0.1, base_speed * 0.1)  # Don't go below 10% of base speed
    max_speed = base_speed * 2.0  # Don't exceed 200% of base speed
    speed_array = np.clip(speed_array, min_speed, max_speed)

    # 5. Smooth the speed profile to remove unrealistic discontinuities
    if len(speed_array) > 50:
        # Use Savitzky-Golay filter for smoothing while preserving features
        window_size = min(51, len(speed_array) // 20 * 2 + 1)  # Odd number
        if window_size >= 5:
            try:
                speed_array = signal.savgol_filter(speed_array, window_size, 3)
            except:
                # Fallback to simple moving average
                window = np.ones(window_size) / window_size
                speed_array = np.convolve(speed_array, window, mode='same')

    # 6. Ensure realistic speed constraints again after smoothing
    speed_array = np.clip(speed_array, min_speed, max_speed)

    # Generate profile statistics
    profile_info = {
        "variation_type": "custom_acceleration",
        "min_speed": np.min(speed_array),
        "max_speed": np.max(speed_array),
        "avg_speed": np.mean(speed_array),
        "speed_std": np.std(speed_array),
        "vehicle_type": vehicle_type,
        "acceleration_events": num_accel_events,
        "gear_changes": num_gear_events,
        "engine_roughness_applied": engine_roughness * 100
    }

    print(f"  Generated profile:")
    print(f"    Speed range: {profile_info['min_speed']:.1f} to {profile_info['max_speed']:.1f} m/s")
    print(f"    Average speed: {profile_info['avg_speed']:.1f} m/s")
    print(f"    Speed variation: ±{profile_info['speed_std']:.1f} m/s")
    print(f"    Acceleration events: {num_accel_events}")
    print(f"    Gear changes: {num_gear_events}")

    return time_array, speed_array, profile_info


def validate_manual_profile_input(time_str, speed_str):
    """
    Validate manual profile input and provide feedback

    Returns:
        tuple: (is_valid, error_message, suggestions)
    """
    try:
        if not time_str or not speed_str:
            return False, "Both time and speed values are required", ["Enter comma-separated values"]

        time_values = [float(x.strip()) for x in time_str.split(',') if x.strip()]
        speed_values = [float(x.strip()) for x in speed_str.split(',') if x.strip()]

        if len(time_values) != len(speed_values):
            return False, "Number of time and speed values must match", [
                f"Time values: {len(time_values)}, Speed values: {len(speed_values)}"
            ]

        if len(time_values) < 2:
            return False, "Need at least 2 time/speed points", [
                "Example: Time: 0,2,5  Speed: 20,30,15"
            ]

        if any(t < 0 for t in time_values):
            return False, "Time values cannot be negative", [
                "Use only positive time values starting from 0"
            ]

        if any(s <= 0 for s in speed_values):
            return False, "Speed values must be positive", [
                "Use speeds > 0 m/s (e.g., 5, 10, 20)"
            ]

        # Check for reasonable values
        max_time = max(time_values)
        max_speed = max(speed_values)

        suggestions = []
        if max_time > 60:
            suggestions.append("Consider shorter simulation times (< 60s)")
        if max_speed > 100:
            suggestions.append("Very high speeds (>100 m/s = 360 km/h)")
        if max_speed < 1:
            suggestions.append("Very low speeds (<1 m/s = 3.6 km/h)")

        return True, "Valid input", suggestions

    except ValueError as e:
        return False, f"Invalid number format: {e}", [
            "Use numbers separated by commas",
            "Example: 0,1.5,3.0,5"
        ]


def get_preset_profile(preset_name):
    """
    Get a preset manual profile

    Args:
        preset_name: Name of the preset

    Returns:
        dict: Preset profile data or None if not found
    """
    return PRESET_MANUAL_PROFILES.get(preset_name)


def get_all_preset_names():
    """
    Get list of all available preset names

    Returns:
        list: List of preset names with descriptions
    """
    return [(name, data["name"], data["description"])
            for name, data in PRESET_MANUAL_PROFILES.items()]


# Integration function for enhanced Doppler calculations
def calculate_enhanced_doppler_effect(path_type, path_params, acceleration_mode,
                                     custom_accel_params, manual_profile_data,
                                     vehicle_type, duration):
    """
    Calculate Doppler effect with enhanced acceleration profile support

    Args:
        path_type: 'straight', 'parabola', or 'bezier'
        path_params: Dictionary of path-specific parameters
        acceleration_mode: 'custom' or 'manual'
        custom_accel_params: Custom acceleration parameters
        manual_profile_data: Manual profile data if mode is 'manual'
        vehicle_type: 'car', 'train', 'flight', or 'drone'
        duration: Audio duration in seconds

    Returns:
        tuple: (freq_ratios, amplitudes, speed_profile_info)
    """
    base_speed = path_params.get('speed', 20)  # Default speed

    print(f"Enhanced Doppler calculation:")
    print(f"  Path: {path_type}")
    print(f"  Acceleration mode: {acceleration_mode}")
    print(f"  Vehicle: {vehicle_type}")
    print(f"  Duration: {duration}s")

    # Generate acceleration profile
    if acceleration_mode == 'manual':
        time_array, speed_array, profile_info = parse_manual_acceleration_profile(
            manual_profile_data['time_values'],
            manual_profile_data['speed_values'],
            duration
        )
    elif acceleration_mode == 'custom':
        time_array, speed_array, profile_info = generate_custom_speed_profile(
            base_speed, duration, vehicle_type, custom_accel_params
        )
    else:
        raise ValueError(f"Unsupported acceleration mode: {acceleration_mode}")

    # Calculate Doppler effect based on path type and acceleration profile
    if acceleration_mode == 'manual':
        # For manual profiles, we override the path speed completely
        freq_ratios, amplitudes = calculate_manual_profile_doppler(
            path_type, path_params, time_array, speed_array, duration
        )
    else:
        # Custom mode - use realistic simulation with variations
        freq_ratios, amplitudes = calculate_realistic_doppler_with_profile(
            path_type, path_params, time_array, speed_array, vehicle_type, duration
        )

    return freq_ratios, amplitudes, profile_info


def calculate_manual_profile_doppler(path_type, path_params, time_array, speed_array, duration):
    """
    Calculate Doppler effect for manual speed profiles

    FIXED: Improved Bezier curve calculations to use proper control points
    """
    c = 343.0  # Speed of sound
    num_points = min(200, len(time_array) // 10)  # Reasonable number of points
    time_indices = np.linspace(0, len(time_array)-1, num_points, dtype=int)

    freq_ratios = []
    amplitudes = []

    print(f"Manual profile Doppler calculation:")
    print(f"  Using {num_points} calculation points")
    print(f"  Speed range: {np.min(speed_array):.1f} to {np.max(speed_array):.1f} m/s")

    # FIXED: Import bezier helper functions for proper Bezier calculations
    if path_type == 'bezier':
        from bezier import bezier_point, bezier_velocity

    for i, time_idx in enumerate(time_indices):
        current_time = time_array[time_idx]
        current_speed = speed_array[time_idx]

        # Calculate position and distance based on path type
        if path_type == 'straight':
            h = path_params.get('h', 10)
            angle = path_params.get('angle', 0)

            # For manual profile, position is integrated speed over time
            position_distance = np.trapz(speed_array[:time_idx+1], time_array[:time_idx+1]) if time_idx > 0 else 0

            # Center the motion around closest approach
            centered_distance = position_distance - np.trapz(speed_array, time_array) / 2

            angle_rad = np.radians(angle)
            x = centered_distance * np.cos(angle_rad)
            y = h + centered_distance * np.sin(angle_rad)

            distance = np.sqrt(x**2 + y**2)
            distance = max(distance, 0.1)

            # Radial velocity
            vx = current_speed * np.cos(angle_rad)
            vy = current_speed * np.sin(angle_rad)
            vr = (vx * x + vy * y) / distance

        elif path_type == 'parabola':
            a = path_params.get('a', 0.1)
            h = path_params.get('h', 10)

            # Position from integrated speed
            position_distance = np.trapz(speed_array[:time_idx+1], time_array[:time_idx+1]) if time_idx > 0 else 0
            centered_distance = position_distance - np.trapz(speed_array, time_array) / 2

            x = centered_distance
            y = a * x**2 + h

            distance = np.sqrt(x**2 + y**2)
            distance = max(distance, 0.1)

            # Velocity components
            vx = current_speed
            vy = 2 * a * x * current_speed
            vr = (vx * x + vy * y) / distance

        elif path_type == 'bezier':
            # FIXED: Use proper Bezier curve calculations with control points
            x0 = path_params.get('x0', -30)
            y0 = path_params.get('y0', 20)
            x1 = path_params.get('x1', -10)
            y1 = path_params.get('y1', -10)
            x2 = path_params.get('x2', 10)
            y2 = path_params.get('y2', -10)
            x3 = path_params.get('x3', 30)
            y3 = path_params.get('y3', 20)

            # Calculate parameter t based on time progression
            t = current_time / duration  # Progress along curve (0 to 1)

            # Use proper Bezier point calculation
            x = bezier_point(t, x0, x1, x2, x3)
            y = bezier_point(t, y0, y1, y2, y3)

            distance = np.sqrt(x**2 + y**2)
            distance = max(distance, 0.1)

            # Calculate velocity direction using Bezier derivative
            vel_x_param = bezier_velocity(t, x0, x1, x2, x3)
            vel_y_param = bezier_velocity(t, y0, y1, y2, y3)

            # Scale velocity to match current speed
            param_speed = np.sqrt(vel_x_param**2 + vel_y_param**2)
            if param_speed > 1e-6:
                scale = current_speed / param_speed
                vx = vel_x_param * scale
                vy = vel_y_param * scale
            else:
                vx = current_speed
                vy = 0

            # Radial velocity
            vr = (vx * x + vy * y) / distance

        # Apply Doppler formula
        vr_limited = np.clip(vr, -c*0.3, c*0.3)
        freq_ratio = c / (c - vr_limited)
        freq_ratios.append(freq_ratio)

        # FIXED: More physical amplitude (inverse distance law)
        amplitude = 1.0 / (distance + 1.0)
        amplitudes.append(amplitude)

    # FIXED: Optional velocity smoothing for manual profiles
    # Apply gentler smoothing to preserve user-specified variations
    if len(freq_ratios) > 10:
        from scipy.signal import savgol_filter
        window_size = min(11, len(freq_ratios) // 10 * 2 + 1)  # Smaller window
        if window_size >= 5:
            try:
                # Gentle smoothing with lower polynomial order
                freq_ratios_smooth = savgol_filter(freq_ratios, window_size, 2)
                # Only apply smoothing if it doesn't change values too much
                max_change = np.max(np.abs(np.array(freq_ratios) - freq_ratios_smooth))
                if max_change < 0.1:  # Less than 10% change
                    freq_ratios = freq_ratios_smooth.tolist()
                    print(f"  Applied gentle smoothing (max change: {max_change:.3f})")
                else:
                    print(f"  Skipped smoothing to preserve manual variations")
            except:
                pass

    print(f"  Frequency ratio range: {min(freq_ratios):.3f} to {max(freq_ratios):.3f}")

    return freq_ratios, amplitudes


def calculate_realistic_doppler_with_profile(path_type, path_params, time_array, speed_array, vehicle_type, duration):
    """
    Calculate Doppler effect using realistic speed profile with custom parameters

    FIXED: Improved Bezier calculations and more physical amplitude
    """
    c = 343.0  # Speed of sound
    num_points = min(200, len(time_array) // 10)
    time_indices = np.linspace(0, len(time_array)-1, num_points, dtype=int)

    freq_ratios = []
    amplitudes = []

    print(f"Realistic Doppler calculation with custom profile:")
    print(f"  Using {num_points} calculation points")
    print(f"  Speed range: {np.min(speed_array):.1f} to {np.max(speed_array):.1f} m/s")

    # FIXED: Import bezier functions if needed
    if path_type == 'bezier':
        from bezier import bezier_point, bezier_velocity

    for i, time_idx in enumerate(time_indices):
        current_time = time_array[time_idx]
        current_speed = speed_array[time_idx]

        # Calculate position based on path type and integrated speed
        if path_type == 'straight':
            h = path_params.get('h', 10)
            angle = path_params.get('angle', 0)

            # Integrate speed to get position
            if time_idx > 0:
                position_distance = np.trapz(speed_array[:time_idx+1], time_array[:time_idx+1])
            else:
                position_distance = 0

            # Center around closest approach
            total_distance = np.trapz(speed_array, time_array)
            centered_distance = position_distance - total_distance / 2

            angle_rad = np.radians(angle)
            x = centered_distance * np.cos(angle_rad)
            y = h + centered_distance * np.sin(angle_rad)

            distance = np.sqrt(x**2 + y**2)
            distance = max(distance, 0.1)

            # Velocity components
            vx = current_speed * np.cos(angle_rad)
            vy = current_speed * np.sin(angle_rad)
            vr = (vx * x + vy * y) / distance

        elif path_type == 'parabola':
            a = path_params.get('a', 0.1)
            h = path_params.get('h', 10)

            # Similar integration approach for parabola
            if time_idx > 0:
                position_distance = np.trapz(speed_array[:time_idx+1], time_array[:time_idx+1])
            else:
                position_distance = 0

            total_distance = np.trapz(speed_array, time_array)
            centered_distance = position_distance - total_distance / 2

            x = centered_distance
            y = a * x**2 + h

            distance = np.sqrt(x**2 + y**2)
            distance = max(distance, 0.1)

            # Velocity components
            vx = current_speed
            vy = 2 * a * x * current_speed
            vr = (vx * x + vy * y) / distance

        elif path_type == 'bezier':
            # FIXED: Proper Bezier calculations
            x0 = path_params.get('x0', -30)
            y0 = path_params.get('y0', 20)
            x1 = path_params.get('x1', -10)
            y1 = path_params.get('y1', -10)
            x2 = path_params.get('x2', 10)
            y2 = path_params.get('y2', -10)
            x3 = path_params.get('x3', 30)
            y3 = path_params.get('y3', 20)

            t = current_time / duration

            # Proper Bezier point calculation
            x = bezier_point(t, x0, x1, x2, x3)
            y = bezier_point(t, y0, y1, y2, y3)

            distance = np.sqrt(x**2 + y**2)
            distance = max(distance, 0.1)

            # Velocity using Bezier derivatives
            vel_x_param = bezier_velocity(t, x0, x1, x2, x3)
            vel_y_param = bezier_velocity(t, y0, y1, y2, y3)

            param_speed = np.sqrt(vel_x_param**2 + vel_y_param**2)
            if param_speed > 1e-6:
                scale = current_speed / param_speed
                vx = vel_x_param * scale
                vy = vel_y_param * scale
            else:
                vx = current_speed
                vy = 0

            vr = (vx * x + vy * y) / distance

        # Apply Doppler formula
        vr_limited = np.clip(vr, -c*0.3, c*0.3)
        freq_ratio = c / (c - vr_limited)
        freq_ratios.append(freq_ratio)

        # FIXED: More physical amplitude with speed-dependent effects
        base_amplitude = 1.0 / (distance + 1.0)  # Inverse distance law

        # Add speed-dependent amplitude variations (engine load effects)
        base_speed = path_params.get('speed', 20)
        speed_factor = current_speed / base_speed  # Normalized speed
        engine_load_factor = 1.0 + 0.1 * (speed_factor - 1.0)  # ±10% based on engine load

        amplitude = base_amplitude * engine_load_factor
        amplitudes.append(amplitude)

    print(f"  Frequency ratio range: {min(freq_ratios):.3f} to {max(freq_ratios):.3f}")

    return freq_ratios, amplitudes


def get_vehicle_profile(vehicle_type):
    """
    Get vehicle profile information

    Args:
        vehicle_type: 'car', 'train', 'flight', or 'drone'

    Returns:
        dict: Vehicle profile with all parameters
    """
    return VEHICLE_PROFILES.get(vehicle_type, VEHICLE_PROFILES['car'])


def validate_speed_profile(speed_array, vehicle_type='car'):
    """
    Validate that a speed profile is realistic for the given vehicle type

    Args:
        speed_array: Array of speeds
        vehicle_type: Vehicle type to validate against

    Returns:
        tuple: (is_valid, warnings, statistics)
    """
    profile = VEHICLE_PROFILES.get(vehicle_type, VEHICLE_PROFILES['car'])
    warnings = []

    min_speed = np.min(speed_array)
    max_speed = np.max(speed_array)
    avg_speed = np.mean(speed_array)
    speed_changes = np.abs(np.diff(speed_array))
    max_speed_change = np.max(speed_changes)

    # Check speed ranges
    typical_min, typical_max = profile['typical_speeds']
    if max_speed > typical_max * 1.5:
        warnings.append(f"Maximum speed ({max_speed:.1f} m/s) is unusually high for {vehicle_type}")

    if min_speed < typical_min * 0.1:
        warnings.append(f"Minimum speed ({min_speed:.1f} m/s) is unusually low for {vehicle_type}")

    # Check acceleration rates
    dt = 0.01  # Assumed time step
    max_acceleration = max_speed_change / dt
    vehicle_max_accel = profile['max_acceleration'][1]

    if max_acceleration > vehicle_max_accel * 2:
        warnings.append(f"Maximum acceleration ({max_acceleration:.1f} m/s²) exceeds typical {vehicle_type} capabilities")

    statistics = {
        'min_speed': min_speed,
        'max_speed': max_speed,
        'avg_speed': avg_speed,
        'speed_std': np.std(speed_array),
        'max_acceleration': max_acceleration,
        'warnings_count': len(warnings)
    }

    is_valid = len(warnings) == 0

    return is_valid, warnings, statistics


if __name__ == "__main__":
    # Test the realistic simulation
    print("Testing Realistic Vehicle Simulation...")
    print("=" * 50)

    # Test different vehicle types (added 'drone')
    for vehicle in ['car', 'train', 'flight', 'drone']:
        print(f"\nTesting {vehicle}:")

        # Set a sensible base speed for drones if default base provided (keep it simple)
        base_speed = 30
        if vehicle == 'drone':
            base_speed = 8  # m/s typical small UAV cruise

        # Generate custom speed profile
        time_arr, speed_arr, info = generate_custom_speed_profile(
            base_speed=base_speed, duration=10, vehicle_type=vehicle,
            custom_params={'max_acceleration': 3.0, 'engine_roughness': 15}
        )

        print(f"Generated {len(speed_arr)} speed samples")

        # Validate the profile
        is_valid, warnings, stats = validate_speed_profile(speed_arr, vehicle)
        print(f"Profile valid: {is_valid}")
        if warnings:
            for warning in warnings:
                print(f"  Warning: {warning}")

    print("\n" + "=" * 50)
    print("Realistic simulation testing complete!")
