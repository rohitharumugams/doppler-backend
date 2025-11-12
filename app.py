# app.py - Fixed & robust Flask API for Doppler simulator (React Native ready)

from flask import Flask, render_template, request, send_file, jsonify
from flask_cors import CORS
import numpy as np
import os
import uuid
from datetime import datetime

# audio utilities (your existing module)
from audio_utils import (
    load_original_audio,
    apply_doppler_to_audio_fixed,
    apply_doppler_to_audio_fixed_alternative,
    apply_doppler_to_audio_fixed_advanced,
    normalize_amplitudes,
    save_audio,
    SR
)

# optional enhanced simulation utilities (may raise)
from realistic_simulation import (
    calculate_enhanced_doppler_effect,
    validate_manual_profile_input,
    get_preset_profile,
    PRESET_MANUAL_PROFILES
)

# 2D path calculators (cars/trains)
from straight_line import calculate_straight_line_doppler
from parabola import calculate_parabola_doppler
from bezier import calculate_bezier_doppler

# 3D path calculators (drone)
from drone_3d import (
    calculate_straight_line_3d_doppler,
    calculate_parabola_3d_doppler,
    calculate_bezier_3d_doppler
)

# -----------------------------------------------------------------------------
# App and config
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # tighten origins in production

# ensure directories
os.makedirs('static', exist_ok=True)
os.makedirs('static/outputs', exist_ok=True)

# simple in-memory job store (replace with DB/Redis in prod)
job_store = {}

# explicit mapping from vehicle id to preferred source sample key/filename
# (audio_utils.load_original_audio should accept the vehicle_id string;
#  if your load_original_audio expects filenames adapt accordingly)
VEHICLE_SOURCE_MAP = {
    'car': 'car_engine_loop.wav',
    'train': 'train_whistle_loop.wav',
    'drone': 'drone_rotor_loop.wav'
}

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _safe_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default

def _load_audio_resilient(vehicle_type, duration):
    """
    Try to load original audio for `vehicle_type` robustly.
    If load_original_audio supports a 'source_name' or filename signature, this
    function will attempt the more explicit call and fall back to the simple call.
    """
    source_filename = VEHICLE_SOURCE_MAP.get(vehicle_type, None)
    # First attempt: assume load_original_audio(vehicle_type, duration)
    try:
        return load_original_audio(vehicle_type, duration)
    except TypeError:
        # maybe load_original_audio expects (filename, duration)
        try:
            if source_filename:
                return load_original_audio(source_filename, duration)
        except Exception:
            pass
    except Exception:
        pass

    # Final fallback: try any mapping filename directly
    try:
        if source_filename:
            return load_original_audio(source_filename, duration)
    except Exception as e:
        # give up and re-raise a clear error
        raise RuntimeError(f"Failed to load original audio for vehicle '{vehicle_type}': {e}")

# -----------------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/info', methods=['GET'])
def api_info():
    return jsonify({
        'version': '2.0-api-fixed',
        'status': 'online',
        'features': {
            'drone_3d': True,
            'manual_profiles': True
        }
    })

@app.route('/api/vehicles', methods=['GET'])
def get_vehicles():
    return jsonify({
        'vehicles': [
            {'id': 'car', 'name': 'Passenger Car', 'description': 'Car engine/horn sound'},
            {'id': 'train', 'name': 'Train', 'description': 'Train horn/whistle'},
            {'id': 'drone', 'name': 'Drone', 'description': 'Drone rotor sound (3D)'},
        ]
    })

@app.route('/api/paths', methods=['GET'])
def get_paths():
    vehicle_type = request.args.get('vehicle_type', 'car')
    # base 2D paths
    base_paths = [
        {
            'id': 'straight',
            'name': 'Straight Line',
            'description': 'Straight line path',
            'parameters': [
                {'name': 'speed', 'type': 'number', 'unit': 'm/s', 'default': 20},
                {'name': 'h', 'type': 'number', 'unit': 'm', 'default': 10},
                {'name': 'angle', 'type': 'number', 'unit': 'deg', 'default': 0}
            ]
        },
        {
            'id': 'parabola',
            'name': 'Parabola',
            'description': 'Parabolic trajectory',
            'parameters': [
                {'name': 'speed', 'type': 'number', 'unit': 'm/s', 'default': 20},
                {'name': 'a', 'type': 'number', 'unit': 'curvature', 'default': 0.1},
                {'name': 'h', 'type': 'number', 'unit': 'm', 'default': 10}
            ]
        },
        {
            'id': 'bezier',
            'name': 'Bezier',
            'description': 'Bezier curve (2D)',
            'parameters': [
                {'name': 'speed', 'type': 'number', 'unit': 'm/s', 'default': 20},
                {'name': 'x0', 'type': 'number', 'unit': 'm', 'default': -30},
                {'name': 'y0', 'type': 'number', 'unit': 'm', 'default': 20},
                {'name': 'x1', 'type': 'number', 'unit': 'm', 'default': -10},
                {'name': 'y1', 'type': 'number', 'unit': 'm', 'default': -10},
                {'name': 'x2', 'type': 'number', 'unit': 'm', 'default': 10},
                {'name': 'y2', 'type': 'number', 'unit': 'm', 'default': -10},
                {'name': 'x3', 'type': 'number', 'unit': 'm', 'default': 30},
                {'name': 'y3', 'type': 'number', 'unit': 'm', 'default': 20},
            ]
        }
    ]

    if vehicle_type == 'drone':
        # produce 3D variations (z coordinates + 3D-specific params)
        paths_3d = [
            {
                'id': 'straight',
                'name': 'Straight Line 3D',
                'description': '3D straight line (drone)',
                'parameters': [
                    {'name': 'speed', 'type': 'number', 'unit': 'm/s', 'default': 20},
                    {'name': 'h', 'type': 'number', 'unit': 'm', 'default': 10},
                    {'name': 'angle_xy', 'type': 'number', 'unit': 'deg', 'default': 0},
                    {'name': 'angle_z', 'type': 'number', 'unit': 'deg', 'default': 0}
                ]
            },
            {
                'id': 'parabola',
                'name': 'Parabola 3D',
                'description': '3D parabola (X-Z plane with Y offset)',
                'parameters': [
                    {'name': 'speed', 'type': 'number', 'unit': 'm/s', 'default': 20},
                    {'name': 'a', 'type': 'number', 'unit': 'curvature', 'default': 0.1},
                    {'name': 'h', 'type': 'number', 'unit': 'm', 'default': 10},
                    {'name': 'z_offset', 'type': 'number', 'unit': 'm', 'default': 10}
                ]
            },
            {
                'id': 'bezier',
                'name': 'Bezier 3D',
                'description': '3D cubic Bezier (drone)',
                'parameters': [
                    {'name': 'speed', 'type': 'number', 'unit': 'm/s', 'default': 20},
                    {'name': 'x0', 'type': 'number', 'unit': 'm', 'default': -30},
                    {'name': 'y0', 'type': 'number', 'unit': 'm', 'default': 10},
                    {'name': 'z0', 'type': 'number', 'unit': 'm', 'default': 10},
                    {'name': 'x1', 'type': 'number', 'unit': 'm', 'default': -10},
                    {'name': 'y1', 'type': 'number', 'unit': 'm', 'default': 20},
                    {'name': 'z1', 'type': 'number', 'unit': 'm', 'default': 20},
                    {'name': 'x2', 'type': 'number', 'unit': 'm', 'default': 10},
                    {'name': 'y2', 'type': 'number', 'unit': 'm', 'default': 20},
                    {'name': 'z2', 'type': 'number', 'unit': 'm', 'default': 20},
                    {'name': 'x3', 'type': 'number', 'unit': 'm', 'default': 30},
                    {'name': 'y3', 'type': 'number', 'unit': 'm', 'default': 10},
                    {'name': 'z3', 'type': 'number', 'unit': 'm', 'default': 10},
                ]
            }
        ]
        return jsonify({'paths': paths_3d})
    else:
        return jsonify({'paths': base_paths})

@app.route('/api/presets', methods=['GET'])
def get_presets():
    # expose simple preset list
    presets = []
    for key, p in PRESET_MANUAL_PROFILES.items():
        presets.append({'id': key, 'name': p['name'], 'duration': p.get('duration', 5)})
    return jsonify({'presets': presets})

@app.route('/api/preset/<preset_name>', methods=['GET'])
def get_preset_detail(preset_name):
    preset = get_preset_profile(preset_name)
    if preset:
        return jsonify(preset)
    return jsonify({'error': 'Preset not found'}), 404

# -----------------------------------------------------------------------------
# Simulation endpoint
# -----------------------------------------------------------------------------
@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        job_id = str(uuid.uuid4())

        # Basic params
        path = data.get('path', 'straight')
        vehicle_type = data.get('vehicle_type', 'car')
        shift_method = data.get('shift_method', 'timestretch')
        audio_duration = _safe_float(data.get('audio_duration', 5.0), 5.0)
        acceleration_mode = data.get('acceleration_mode', 'perfect')

        # Persist job
        job_store[job_id] = {
            'status': 'processing',
            'progress': 0,
            'created_at': datetime.now().isoformat(),
            'parameters': data
        }

        # Immediately process synchronously (for now)
        try:
            result = process_simulation(job_id, path, vehicle_type, shift_method, audio_duration, acceleration_mode, data)
            job_store[job_id].update({
                'status': 'completed',
                'progress': 100,
                'result': result,
                'completed_at': datetime.now().isoformat()
            })
        except Exception as e:
            job_store[job_id].update({
                'status': 'failed',
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            })
            raise

        return jsonify({
            'job_id': job_id,
            'status': job_store[job_id]['status'],
            'result': job_store[job_id]['result']
        }), 200

    except Exception as e:
        # Log and return error
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# -----------------------------------------------------------------------------
# Core processing routine
# -----------------------------------------------------------------------------
def process_simulation(job_id, path, vehicle_type, shift_method, audio_duration, acceleration_mode, data):
    """
    Compute Doppler frequency ratios and amplitudes, apply shift, and save output.
    Returns metadata dictionary for client.
    """
    # 1) Load original audio (resiliently)
    original_audio = _load_audio_resilient(vehicle_type, audio_duration)

    # 2) Build path_params with safe defaults and parse numeric values
    path_params = {}

    # Generic defaults
    path_params['speed'] = _safe_float(data.get('speed', 20.0), 20.0)

    if path == 'straight':
        path_params['h'] = _safe_float(data.get('h', 10.0), 10.0)
        # For 2D path angle param remains 'angle'
        path_params['angle'] = _safe_float(data.get('angle', 0.0), 0.0)
    elif path == 'parabola':
        path_params['a'] = _safe_float(data.get('a', 0.1), 0.1)
        path_params['h'] = _safe_float(data.get('h', 10.0), 10.0)
    elif path == 'bezier':
        # 2D bezier required parameters (x,y)
        path_params.update({
            'x0': _safe_float(data.get('x0', -30.0), -30.0),
            'x1': _safe_float(data.get('x1', -10.0), -10.0),
            'x2': _safe_float(data.get('x2', 10.0), 10.0),
            'x3': _safe_float(data.get('x3', 30.0), 30.0),
            'y0': _safe_float(data.get('y0', 20.0), 20.0),
            'y1': _safe_float(data.get('y1', -10.0), -10.0),
            'y2': _safe_float(data.get('y2', -10.0), -10.0),
            'y3': _safe_float(data.get('y3', 20.0), 20.0),
        })

    # 3) If drone, include 3D params (z's and 3D angles)
    if vehicle_type == 'drone':
        if path == 'straight':
            path_params['h'] = _safe_float(data.get('h', path_params.get('h', 10.0)), 10.0)
            path_params['angle_xy'] = _safe_float(data.get('angle_xy', 0.0), 0.0)
            path_params['angle_z'] = _safe_float(data.get('angle_z', 0.0), 0.0)
        elif path == 'parabola':
            path_params['a'] = _safe_float(data.get('a', path_params.get('a', 0.1)), 0.1)
            path_params['h'] = _safe_float(data.get('h', path_params.get('h', 10.0)), 10.0)
            path_params['z_offset'] = _safe_float(data.get('z_offset', 10.0), 10.0)
        elif path == 'bezier':
            # Ensure y values exist (kept from earlier) and add z values
            for k in ('x0', 'x1', 'x2', 'x3', 'y0', 'y1', 'y2', 'y3'):
                if k not in path_params:
                    path_params[k] = _safe_float(data.get(k, 0.0), 0.0)
            path_params.update({
                'z0': _safe_float(data.get('z0', 10.0), 10.0),
                'z1': _safe_float(data.get('z1', 20.0), 20.0),
                'z2': _safe_float(data.get('z2', 20.0), 20.0),
                'z3': _safe_float(data.get('z3', 10.0), 10.0),
            })

    # 4) Optionally handle enhanced/custom/manual acceleration profiles
    try:
        if acceleration_mode in ['custom', 'manual']:
            # Delegate heavy lifting to realistic_simulation if available
            freq_ratios, amplitudes, extra = calculate_enhanced_doppler_effect(
                path, path_params, acceleration_mode, None, None, vehicle_type, audio_duration
            )
        else:
            # Perfect physics branch
            if vehicle_type == 'drone':
                # ALWAYS use 3D functions for drone
                if path == 'straight':
                    freq_ratios, amplitudes = calculate_straight_line_3d_doppler(
                        path_params['speed'],
                        path_params['h'],
                        path_params.get('angle_xy', 0.0),
                        path_params.get('angle_z', 0.0),
                        audio_duration
                    )
                elif path == 'parabola':
                    freq_ratios, amplitudes = calculate_parabola_3d_doppler(
                        path_params['speed'],
                        path_params['a'],
                        path_params['h'],
                        path_params.get('z_offset', 10.0),
                        audio_duration
                    )
                elif path == 'bezier':
                    freq_ratios, amplitudes = calculate_bezier_3d_doppler(
                        path_params['speed'],
                        path_params['x0'], path_params['x1'], path_params['x2'], path_params['x3'],
                        path_params['y0'], path_params['y1'], path_params['y2'], path_params['y3'],
                        path_params['z0'], path_params['z1'], path_params['z2'], path_params['z3'],
                        audio_duration
                    )
                else:
                    raise ValueError(f"Unknown path '{path}' for drone.")
            else:
                # non-drone: use existing 2D functions
                if path == 'straight':
                    freq_ratios, amplitudes = calculate_straight_line_doppler(
                        path_params['speed'],
                        path_params['h'],
                        path_params.get('angle', 0.0),
                        audio_duration
                    )
                elif path == 'parabola':
                    freq_ratios, amplitudes = calculate_parabola_doppler(
                        path_params['speed'],
                        path_params['a'],
                        path_params['h'],
                        audio_duration
                    )
                elif path == 'bezier':
                    freq_ratios, amplitudes = calculate_bezier_doppler(
                        path_params['speed'],
                        path_params['x0'], path_params['x1'], path_params['x2'], path_params['x3'],
                        path_params['y0'], path_params['y1'], path_params['y2'], path_params['y3'],
                        audio_duration
                    )
                else:
                    raise ValueError(f"Unknown path '{path}' for vehicle '{vehicle_type}'.")
    except Exception as e:
        # If enhanced calculation fails or any internal failure in 3D/2D functions,
        # fallback to a safe straight-line calculation (2D or 3D depending on vehicle)
        import traceback
        traceback.print_exc()
        print("Fallback: using safe straight-line calculation.")

        if vehicle_type == 'drone':
            freq_ratios, amplitudes = calculate_straight_line_3d_doppler(
                path_params.get('speed', 20.0),
                path_params.get('h', 10.0),
                path_params.get('angle_xy', 0.0),
                path_params.get('angle_z', 0.0),
                audio_duration
            )
        else:
            freq_ratios, amplitudes = calculate_straight_line_doppler(
                path_params.get('speed', 20.0),
                path_params.get('h', 10.0),
                path_params.get('angle', 0.0),
                audio_duration
            )

    # 5) Normalize amplitude envelope
    amplitudes = normalize_amplitudes(amplitudes)

    # 6) Ensure freq_ratios & amplitudes are lists or numpy arrays
    freq_ratios = list(freq_ratios) if not isinstance(freq_ratios, list) else freq_ratios
    amplitudes = list(amplitudes) if not isinstance(amplitudes, list) else amplitudes

    # 7) Apply Doppler effect using selected shift method
    if shift_method == 'resample':
        doppler_audio = apply_doppler_to_audio_fixed_alternative(original_audio, freq_ratios, amplitudes)
    elif shift_method == 'advanced':
        doppler_audio = apply_doppler_to_audio_fixed_advanced(original_audio, freq_ratios, amplitudes)
    else:
        doppler_audio = apply_doppler_to_audio_fixed(original_audio, freq_ratios, amplitudes)

    # 8) Trim or pad to desired duration in samples
    target_samples = int(SR * audio_duration)
    doppler_audio = np.asarray(doppler_audio, dtype=np.float32)

    if doppler_audio.shape[0] > target_samples:
        doppler_audio = doppler_audio[:target_samples]
    elif doppler_audio.shape[0] < target_samples:
        padded = np.zeros(target_samples, dtype=np.float32)
        padded[: doppler_audio.shape[0]] = doppler_audio
        doppler_audio = padded

    # 9) Save WAV file
    filename = f"{job_id}.wav"
    output_path = os.path.join('static', 'outputs', filename)
    saved_duration = save_audio(doppler_audio, output_path)

    # 10) Prepare result metadata
    result = {
        'filename': filename,
        'duration': saved_duration,
        'download_url': f'/api/download/{filename}',
        'freq_ratio_range': {
            'min': float(np.min(freq_ratios)),
            'max': float(np.max(freq_ratios))
        }
    }
    return result

# -----------------------------------------------------------------------------
# Job status, download, health and helpers
# -----------------------------------------------------------------------------
@app.route('/api/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    job = job_store.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    filepath = os.path.join('static', 'outputs', filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    return send_file(filepath, as_attachment=True, download_name=filename)

@app.route('/health', methods=['GET'])
def health():
    active = sum(1 for j in job_store.values() if j.get('status') == 'processing')
    return jsonify({'status': 'healthy', 'active_jobs': active, 'version': '2.0-api-fixed'})

# optional cleanup helper (call from scheduler)
def cleanup_old_files(max_age_hours=24):
    import time
    output_dir = os.path.join('static', 'outputs')
    now = time.time()
    for fn in os.listdir(output_dir):
        fp = os.path.join(output_dir, fn)
        try:
            age_hours = (now - os.path.getmtime(fp)) / 3600.0
            if age_hours > max_age_hours:
                os.remove(fp)
        except Exception as e:
            print(f"Cleanup error for {fp}: {e}")

# -----------------------------------------------------------------------------
# Run server
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print("Starting Doppler Simulator (fixed app.py)")
    cleanup_old_files()  # light cleanup on startup
    app.run(debug=True, host='0.0.0.0', port=5050)
