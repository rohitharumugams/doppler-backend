# app.py - API-ready Flask application for React Native
# Modified for cross-platform mobile app communication

from flask import Flask, render_template, request, send_file, jsonify
from flask_cors import CORS
import numpy as np
import os
import uuid
import time
from datetime import datetime

from audio_utils import (
    load_original_audio,
    apply_doppler_to_audio_fixed, 
    apply_doppler_to_audio_fixed_alternative,
    apply_doppler_to_audio_fixed_advanced,
    normalize_amplitudes, 
    save_audio,
    SR
)

# Import manual acceleration profile support
from realistic_simulation import (
    calculate_enhanced_doppler_effect,
    validate_manual_profile_input,
    get_preset_profile
)

# Import original path calculations
from straight_line import calculate_straight_line_doppler
from parabola import calculate_parabola_doppler
from bezier import calculate_bezier_doppler

app = Flask(__name__)

# Enable CORS for React Native communication
CORS(app, resources={
    r"/*": {
        "origins": "*",  # In production, replace with your app's domain
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Store for temporary job results (in production, use Redis or database)
job_store = {}

# Ensure directories exist
os.makedirs('static', exist_ok=True)
os.makedirs('static/outputs', exist_ok=True)

@app.route('/')
def home():
    """Web interface (optional - keep for testing)"""
    return render_template('index.html')

@app.route('/api/info', methods=['GET'])
def api_info():
    """Get API information and available options"""
    return jsonify({
        'version': '2.0-api',
        'status': 'online',
        'endpoints': {
            'simulate': '/api/simulate',
            'job_status': '/api/job/<job_id>',
            'download': '/api/download/<filename>',
            'vehicle_types': '/api/vehicles',
            'path_types': '/api/paths',
            'presets': '/api/presets'
        },
        'features': {
            'manual_profiles': True,
            'custom_acceleration': True,
            'perfect_physics': True,
            'multiple_paths': True,
            'multiple_vehicles': True,
            'multiple_audio_methods': True
        }
    })

@app.route('/api/vehicles', methods=['GET'])
def get_vehicles():
    """Get available vehicle types"""
    return jsonify({
        'vehicles': [
            {
                'id': 'car',
                'name': 'Passenger Car',
                'description': 'Standard car horn sound'
            },
            {
                'id': 'train',
                'name': 'Heavy Train',
                'description': 'Train horn/whistle sound'
            }
        ]
    })

@app.route('/api/paths', methods=['GET'])
def get_paths():
    """Get available path types and their parameters"""
    vehicle_type = request.args.get('vehicle_type', 'car')
    
    all_paths = [
        {
            'id': 'straight',
            'name': 'Straight Line',
            'description': 'Vehicle moves in a straight line past observer',
            'parameters': [
                {'name': 'speed', 'type': 'number', 'unit': 'm/s', 'default': 20, 'min': 1, 'max': 100},
                {'name': 'h', 'type': 'number', 'unit': 'm', 'default': 10, 'min': 1, 'max': 100},
                {'name': 'angle', 'type': 'number', 'unit': 'degrees', 'default': 0, 'min': -45, 'max': 45}
            ]
        },
        {
            'id': 'parabola',
            'name': 'Parabolic Path',
            'description': 'Vehicle follows a parabolic trajectory',
            'parameters': [
                {'name': 'speed', 'type': 'number', 'unit': 'm/s', 'default': 20, 'min': 1, 'max': 100},
                {'name': 'a', 'type': 'number', 'unit': 'curvature', 'default': 0.1, 'min': 0.01, 'max': 1},
                {'name': 'h', 'type': 'number', 'unit': 'm', 'default': 10, 'min': 1, 'max': 100}
            ]
        },
        {
            'id': 'bezier',
            'name': 'Bezier Curve',
            'description': 'Vehicle follows a custom Bezier curve',
            'parameters': [
                {'name': 'speed', 'type': 'number', 'unit': 'm/s', 'default': 20, 'min': 1, 'max': 100},
                {'name': 'x0', 'type': 'number', 'unit': 'm', 'default': -30},
                {'name': 'y0', 'type': 'number', 'unit': 'm', 'default': 20},
                {'name': 'x1', 'type': 'number', 'unit': 'm', 'default': -10},
                {'name': 'y1', 'type': 'number', 'unit': 'm', 'default': -10},
                {'name': 'x2', 'type': 'number', 'unit': 'm', 'default': 10},
                {'name': 'y2', 'type': 'number', 'unit': 'm', 'default': -10},
                {'name': 'x3', 'type': 'number', 'unit': 'm', 'default': 30},
                {'name': 'y3', 'type': 'number', 'unit': 'm', 'default': 20}
            ]
        }
    ]
    
    if vehicle_type == 'train':
        filtered_paths = [p for p in all_paths if p['id'] == 'straight']
    else:
        filtered_paths = all_paths
    
    return jsonify({
        'paths': filtered_paths
    })

@app.route('/api/presets', methods=['GET'])
def get_presets():
    """Get preset manual profiles"""
    from realistic_simulation import PRESET_MANUAL_PROFILES
    
    presets = []
    for key, preset in PRESET_MANUAL_PROFILES.items():
        presets.append({
            'id': key,
            'name': preset['name'],
            'description': preset['description'],
            'duration': preset['duration']
        })
    
    return jsonify({'presets': presets})

@app.route('/api/preset/<preset_name>', methods=['GET'])
def get_preset_detail(preset_name):
    """Get specific preset details"""
    preset = get_preset_profile(preset_name)
    if preset:
        return jsonify(preset)
    else:
        return jsonify({'error': 'Preset not found'}), 404

@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    """
    Main simulation endpoint for React Native
    Accepts JSON or form-data
    Returns job ID for async processing
    """
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Extract parameters with defaults
        path = data.get('path', 'straight')
        vehicle_type = data.get('vehicle_type', 'car')
        shift_method = data.get('shift_method', 'timestretch')
        audio_duration = float(data.get('audio_duration', 5))
        acceleration_mode = data.get('acceleration_mode', 'perfect')
        
        print(f"=== API Simulation Request ===")
        print(f"Job ID: {job_id}")
        print(f"Path: {path}")
        print(f"Vehicle: {vehicle_type}")
        print(f"Duration: {audio_duration}s")
        print(f"Acceleration: {acceleration_mode}")
        
        # Initialize job in store
        job_store[job_id] = {
            'status': 'processing',
            'progress': 0,
            'created_at': datetime.now().isoformat(),
            'parameters': data
        }
        
        # Process simulation in background (in production, use Celery/RQ)
        try:
            result = process_simulation(
                job_id, path, vehicle_type, shift_method, 
                audio_duration, acceleration_mode, data
            )
            
            # Update job store with results
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
            print(f"Simulation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Return job ID immediately
        return jsonify({
            'job_id': job_id,
            'status': 'processing',
            'message': 'Simulation started',
            'check_status_url': f'/api/job/{job_id}'
        }), 202
        
    except Exception as e:
        print(f"Error creating simulation job: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def process_simulation(job_id, path, vehicle_type, shift_method, audio_duration, acceleration_mode, data):
    """Process the actual simulation (can be moved to background worker)"""
    
    # Get custom acceleration parameters
    custom_accel_params = None
    if acceleration_mode == 'custom':
        custom_accel_params = {}
        param_names = ['max_acceleration', 'max_deceleration', 'acceleration_frequency',
                      'acceleration_strength', 'gear_change_frequency', 'engine_roughness']
        
        for param in param_names:
            if param in data and data[param]:
                custom_accel_params[param] = float(data[param])
        
        if not custom_accel_params:
            custom_accel_params = None
    
    # Get manual profile data
    manual_profile_data = None
    if acceleration_mode == 'manual':
        time_values = data.get('manual_time_values', '').strip()
        speed_values = data.get('manual_speed_values', '').strip()
        
        if time_values and speed_values:
            is_valid, error_msg, suggestions = validate_manual_profile_input(time_values, speed_values)
            
            if not is_valid:
                raise ValueError(f"Manual profile error: {error_msg}")
            
            manual_profile_data = {
                'time_values': time_values,
                'speed_values': speed_values
            }
    
    # Load original audio
    original_audio = load_original_audio(vehicle_type, audio_duration)
    
    # Prepare path parameters
    path_params = {'speed': 20}
    
    if path == 'straight':
        path_params.update({
            'speed': float(data.get('speed', 20)),
            'h': float(data.get('h', 10)),
            'angle': float(data.get('angle', 0))
        })
    elif path == 'parabola':
        path_params.update({
            'speed': float(data.get('speed', 20)),
            'a': float(data.get('a', 0.1)),
            'h': float(data.get('h', 10))
        })
    elif path == 'bezier':
        path_params.update({
            'speed': float(data.get('speed', 20)),
            'x0': float(data.get('x0', -30)),
            'x1': float(data.get('x1', -10)),
            'x2': float(data.get('x2', 10)),
            'x3': float(data.get('x3', 30)),
            'y0': float(data.get('y0', 20)),
            'y1': float(data.get('y1', -10)),
            'y2': float(data.get('y2', -10)),
            'y3': float(data.get('y3', 20))
        })
    
    # Calculate Doppler effect
    try:
        if acceleration_mode in ['custom', 'manual']:
            freq_ratios, amplitudes, speed_profile_info = calculate_enhanced_doppler_effect(
                path, path_params, acceleration_mode, 
                custom_accel_params, manual_profile_data, vehicle_type, audio_duration
            )
        else:
            # Perfect physics mode
            if path == 'straight':
                freq_ratios, amplitudes = calculate_straight_line_doppler(
                    path_params['speed'], path_params['h'], 
                    path_params.get('angle', 0), audio_duration
                )
            elif path == 'parabola':
                freq_ratios, amplitudes = calculate_parabola_doppler(
                    path_params['speed'], path_params['a'], 
                    path_params['h'], audio_duration
                )
            elif path == 'bezier':
                freq_ratios, amplitudes = calculate_bezier_doppler(
                    path_params['speed'], path_params['x0'], path_params['x1'], 
                    path_params['x2'], path_params['x3'], path_params['y0'], 
                    path_params['y1'], path_params['y2'], path_params['y3'], audio_duration
                )
    except Exception as e:
        print(f"Enhanced calculation failed: {e}, falling back to perfect physics")
        
        if path == 'straight':
            freq_ratios, amplitudes = calculate_straight_line_doppler(
                path_params['speed'], path_params['h'], 
                path_params.get('angle', 0), audio_duration
            )
        elif path == 'parabola':
            freq_ratios, amplitudes = calculate_parabola_doppler(
                path_params['speed'], path_params['a'], 
                path_params['h'], audio_duration
            )
        elif path == 'bezier':
            freq_ratios, amplitudes = calculate_bezier_doppler(
                path_params['speed'], path_params['x0'], path_params['x1'], 
                path_params['x2'], path_params['x3'], path_params['y0'], 
                path_params['y1'], path_params['y2'], path_params['y3'], audio_duration
            )
    
    # Normalize amplitudes
    amplitudes = normalize_amplitudes(amplitudes)
    
    # Apply Doppler effect
    if shift_method == 'resample':
        doppler_audio = apply_doppler_to_audio_fixed_alternative(original_audio, freq_ratios, amplitudes)
    elif shift_method == 'advanced':
        doppler_audio = apply_doppler_to_audio_fixed_advanced(original_audio, freq_ratios, amplitudes)
    else:
        doppler_audio = apply_doppler_to_audio_fixed(original_audio, freq_ratios, amplitudes)
    
    # Ensure correct length
    target_samples = int(SR * audio_duration)
    if len(doppler_audio) != target_samples:
        if len(doppler_audio) > target_samples:
            doppler_audio = doppler_audio[:target_samples]
        else:
            padded = np.zeros(target_samples)
            padded[:len(doppler_audio)] = doppler_audio
            doppler_audio = padded
    
    # Save audio file with unique name
    filename = f"{job_id}.wav"
    output_path = f'static/outputs/{filename}'
    duration = save_audio(doppler_audio, output_path)
    
    print(f"Simulation complete: {output_path}")
    
    return {
        'filename': filename,
        'duration': duration,
        'download_url': f'/api/download/{filename}',
        'freq_ratio_range': {
            'min': float(min(freq_ratios)),
            'max': float(max(freq_ratios))
        }
    }

@app.route('/api/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Check job status"""
    job = job_store.get(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(job)

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download generated audio file"""
    filepath = f'static/outputs/{filename}'
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(filepath, as_attachment=True, download_name=filename)

@app.route('/simulate', methods=['POST'])
def simulate():
    """Legacy web interface endpoint (keep for backward compatibility)"""
    try:
        # Get user inputs
        path = request.form['path']
        vehicle_type = request.form.get('vehicle_type', 'car')
        shift_method = request.form.get('shift_method', 'timestretch')
        audio_duration = float(request.form.get('audio_duration', 5))
        acceleration_mode = request.form.get('acceleration_mode', 'perfect')
        
        print(f"=== Web Doppler Simulation ===")
        print(f"Path: {path}")
        print(f"Vehicle: {vehicle_type}")
        print(f"Duration: {audio_duration}s")
        
        # [Keep all existing simulation logic from original app.py]
        # ... (same as before)
        
        # For brevity, reuse the process_simulation function
        job_id = 'web-' + str(uuid.uuid4())
        result = process_simulation(
            job_id, path, vehicle_type, shift_method,
            audio_duration, acceleration_mode, request.form.to_dict()
        )
        
        output_path = f'static/outputs/{result["filename"]}'
        return send_file(output_path, as_attachment=True)
        
    except Exception as e:
        print(f"Error in web simulation: {e}")
        import traceback
        traceback.print_exc()
        return f"Simulation error: {str(e)}", 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '2.0-api',
        'timestamp': datetime.now().isoformat(),
        'active_jobs': len([j for j in job_store.values() if j['status'] == 'processing'])
    })

# Cleanup old files periodically (run this with a scheduler in production)
def cleanup_old_files(max_age_hours=24):
    """Remove files older than max_age_hours"""
    import time
    
    output_dir = 'static/outputs'
    current_time = time.time()
    
    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        file_age_hours = (current_time - os.path.getmtime(filepath)) / 3600
        
        if file_age_hours > max_age_hours:
            try:
                os.remove(filepath)
                print(f"Cleaned up old file: {filename}")
            except Exception as e:
                print(f"Error cleaning up {filename}: {e}")

if __name__ == '__main__':
    print("=== Doppler Simulator API Starting ===")
    print("Version: 2.0-API (React Native Ready)")
    print("\nAPI Endpoints:")
    print("  GET  /api/info - API information")
    print("  GET  /api/vehicles - Available vehicles")
    print("  GET  /api/paths - Available paths")
    print("  POST /api/simulate - Start simulation")
    print("  GET  /api/job/<id> - Check job status")
    print("  GET  /api/download/<file> - Download result")
    print("  GET  /health - Health check")
    print("\nCORS enabled for React Native")
    print("=" * 50)
    
    # Run cleanup on startup
    cleanup_old_files()
    
    app.run(debug=True, host='0.0.0.0', port=5050)