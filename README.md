# Doppler Effect Simulator - Backend API

A Flask-based REST API that generates realistic acoustic simulations of the Doppler effect for moving vehicles. This backend processes physics calculations and generates audio files with accurate frequency shifts based on vehicle motion paths.

## Overview

This backend serves as the computational engine for the Doppler Effect Simulator mobile application. It handles complex acoustic physics calculations, audio processing, and provides a RESTful API for integration with React Native and web clients.

## Features

- **Multiple Vehicle Types**: Car, train, drone with realistic acoustic profiles
- **Three Motion Paths**: 
  - Straight line with adjustable angle
  - Parabolic trajectories
  - Cubic Bézier curves
- **Advanced Physics**:
  - Accurate Doppler frequency shift calculations
  - Inverse-square law amplitude modeling
  - Radial velocity computation
- **Multiple Audio Processing Methods**:
  - Time-stretch resampling (default)
  - Spectral domain processing
  - Phase modulation approach
- **Realistic Vehicle Profiles**: Configurable acceleration, deceleration, and engine characteristics
- **Asynchronous Processing**: Job-based system for handling long-running simulations
- **CORS-Enabled**: Ready for cross-origin requests from mobile apps

## Technology Stack

- **Framework**: Flask 2.3.0
- **Audio Processing**: 
  - librosa 0.10.0 (audio analysis and manipulation)
  - soundfile 0.12.1 (audio I/O)
  - numpy 1.24.3 (numerical computations)
  - scipy 1.10.1 (signal processing)
- **API**: Flask-CORS 4.0.0
- **Deployment**: Gunicorn 21.2.0

## Project Structure

```
backend/
├── app.py                          # Main Flask application with API endpoints
├── audio_utils.py                  # Audio loading, processing, and Doppler effect application
├── straight_line.py                # Straight line motion physics
├── parabola.py                     # Parabolic motion physics  
├── bezier.py                       # Bézier curve motion physics
├── realistic_simulation.py         # Advanced vehicle simulation with acceleration profiles
├── requirements.txt                # Python dependencies
├── static/                         # Static assets
│   ├── horn.mp3                   # Car horn audio
│   ├── train.mp3                  # Train horn audio
│   ├── drone.mp3                  # Drone rotor audio
│   └── outputs/                   # Generated audio files
└── templates/
    └── index.html                 # Optional web interface
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- FFmpeg (for audio processing - installed with librosa)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd doppler-simulator-backend
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Prepare audio files:
   - Place vehicle audio files in `static/` directory:
     - `horn.mp3` (car horn)
     - `train.mp3` (train horn/whistle)
     - `drone.mp3` (drone rotor sound)
   - Or use automatic fallback generation

5. Start the development server:
```bash
python app.py
```

The server will start on `http://0.0.0.0:5050`

## API Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0-api",
  "timestamp": "2025-01-20T10:30:00",
  "active_jobs": 0
}
```

### Get API Information
```http
GET /api/info
```

**Response:**
```json
{
  "version": "2.0-api",
  "status": "online",
  "endpoints": {
    "simulate": "/api/simulate",
    "job_status": "/api/job/<job_id>",
    "download": "/api/download/<filename>"
  },
  "features": {
    "manual_profiles": true,
    "custom_acceleration": true,
    "perfect_physics": true,
    "multiple_paths": true,
    "multiple_vehicles": true
  }
}
```

### Get Available Vehicles
```http
GET /api/vehicles
```

**Response:**
```json
{
  "vehicles": [
    {
      "id": "car",
      "name": "Passenger Car",
      "description": "Standard car horn sound"
    },
    {
      "id": "train",
      "name": "Heavy Train",
      "description": "Train horn/whistle sound"
    },
    {
      "id": "drone",
      "name": "Multirotor Drone",
      "description": "Drone rotor buzz sound"
    }
  ]
}
```

### Get Available Paths
```http
GET /api/paths?vehicle_type=car
```

**Response:**
```json
{
  "paths": [
    {
      "id": "straight",
      "name": "Straight Line",
      "description": "Vehicle moves in a straight line past observer",
      "parameters": [
        {"name": "speed", "type": "number", "unit": "m/s", "default": 20, "min": 1, "max": 100},
        {"name": "h", "type": "number", "unit": "m", "default": 10, "min": 1, "max": 100},
        {"name": "angle", "type": "number", "unit": "degrees", "default": 0, "min": -45, "max": 45}
      ]
    }
  ]
}
```

### Start Simulation
```http
POST /api/simulate
Content-Type: application/json
```

**Request Body:**

*Straight Line:*
```json
{
  "path": "straight",
  "vehicle_type": "car",
  "speed": 30,
  "h": 10,
  "angle": 0,
  "audio_duration": 5,
  "shift_method": "timestretch"
}
```

*Parabolic Path:*
```json
{
  "path": "parabola",
  "vehicle_type": "car",
  "speed": 25,
  "a": 0.1,
  "h": 15,
  "audio_duration": 5
}
```

*Bézier Curve:*
```json
{
  "path": "bezier",
  "vehicle_type": "drone",
  "speed": 20,
  "x0": -30, "y0": 20,
  "x1": -10, "y1": -10,
  "x2": 10, "y2": -10,
  "x3": 30, "y3": 20,
  "audio_duration": 5
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "message": "Simulation started"
}
```

### Check Job Status
```http
GET /api/job/{job_id}
```

**Response (Processing):**
```json
{
  "status": "processing",
  "progress": 50,
  "created_at": "2025-01-20T10:30:00"
}
```

**Response (Completed):**
```json
{
  "status": "completed",
  "progress": 100,
  "result": {
    "filename": "550e8400-e29b-41d4-a716-446655440000.wav",
    "duration": 5.0,
    "download_url": "/api/download/550e8400-e29b-41d4-a716-446655440000.wav",
    "freq_ratio_range": {
      "min": 0.923,
      "max": 1.089
    }
  }
}
```

**Response (Failed):**
```json
{
  "status": "failed",
  "error": "Error message describing the failure"
}
```

### Download Audio File
```http
GET /api/download/{filename}
```

**Response:** 
- Content-Type: `audio/wav`
- Binary WAV audio file

## Physics Calculations

### Doppler Effect Formula

The frequency shift is calculated using:

```
f' = f₀ × (c / (c - vᵣ))
```

Where:
- `f'` = observed frequency
- `f₀` = source frequency
- `c` = speed of sound (343 m/s)
- `vᵣ` = radial velocity (positive when approaching)

### Amplitude Calculation

Sound intensity follows the inverse-square law:

```
amplitude = 1.0 / (distance + 1.0)
```

With additional directional factors for realistic acoustic modeling.

### Straight Line Motion

For a vehicle moving in a straight line:

```python
# Position at time t
x(t) = speed × t × cos(angle)
y(t) = h + x(t) × tan(angle)

# Distance from observer at origin
r(t) = √(x² + y²)

# Radial velocity
vᵣ = (vₓ × x + vᵧ × y) / r
```

### Parabolic Motion

For parabolic trajectories:

```python
# Parabola equation: y = a×x² + h
x(t) = speed × t
y(t) = a × x² + h

# Vertical velocity
vᵧ = 2 × a × x × speed
```

### Bézier Curve Motion

Cubic Bézier curve defined by 4 control points (P₀, P₁, P₂, P₃):

```python
# Position at parameter t ∈ [0,1]
P(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃

# Velocity (derivative)
P'(t) = 3(1-t)²(P₁-P₀) + 6(1-t)t(P₂-P₁) + 3t²(P₃-P₂)
```

## Audio Processing Methods

### 1. Time-Stretch Resampling (Default)
- Uses librosa's time stretching with variable pitch
- Best balance of quality and computational efficiency
- Handles both compression and expansion of audio

### 2. Spectral Domain Processing
- STFT-based frequency shifting
- Precise frequency control
- Good for extreme Doppler shifts

### 3. Phase Modulation
- Direct phase trajectory manipulation
- Creates smooth frequency sweeps
- Advanced method for research applications

## Configuration

### Environment Variables

```bash
# Server configuration
FLASK_HOST=0.0.0.0
FLASK_PORT=5050
FLASK_DEBUG=True

# Audio settings
SAMPLE_RATE=22050
MAX_AUDIO_DURATION=30

# File cleanup
MAX_FILE_AGE_HOURS=24
```

### Vehicle Profiles

Modify `realistic_simulation.py` to customize vehicle characteristics:

```python
VEHICLE_PROFILES = {
    'car': {
        'max_acceleration': (2.0, 4.0),  # m/s²
        'max_deceleration': (3.0, 8.0),
        'typical_speeds': (5, 60),
        'mass': 1500,  # kg
        'drag_coefficient': 0.3
    }
}
```

## Production Deployment

### Using Gunicorn

```bash
gunicorn -w 4 -b 0.0.0.0:5050 app:app
```

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5050
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5050", "app:app"]
```

Build and run:
```bash
docker build -t doppler-simulator-backend .
docker run -p 5050:5050 doppler-simulator-backend
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name doppler-simulator.duckdns.org;

    location / {
        proxy_pass http://localhost:5050;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### HTTPS with Let's Encrypt

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d doppler-simulator.duckdns.org

# Auto-renewal
sudo certbot renew --dry-run
```

## Testing

### Manual API Testing

Using curl:
```bash
# Health check
curl http://localhost:5050/health

# Get vehicles
curl http://localhost:5050/api/vehicles

# Start simulation
curl -X POST http://localhost:5050/api/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "path": "straight",
    "vehicle_type": "car",
    "speed": 30,
    "h": 10,
    "angle": 0,
    "audio_duration": 5
  }'

# Check job status (replace with actual job_id)
curl http://localhost:5050/api/job/{job_id}

# Download result
curl -O http://localhost:5050/api/download/{filename}
```

Using Python:
```python
import requests

# Start simulation
response = requests.post('http://localhost:5050/api/simulate', json={
    'path': 'bezier',
    'vehicle_type': 'drone',
    'speed': 20,
    'x0': -30, 'y0': 20,
    'x1': -10, 'y1': -10,
    'x2': 10, 'y2': -10,
    'x3': 30, 'y3': 20,
    'audio_duration': 5
})

job_id = response.json()['job_id']
print(f"Job started: {job_id}")

# Poll for completion
import time
while True:
    status = requests.get(f'http://localhost:5050/api/job/{job_id}').json()
    print(f"Status: {status['status']}")
    
    if status['status'] == 'completed':
        download_url = status['result']['download_url']
        print(f"Download: http://localhost:5050{download_url}")
        break
    
    time.sleep(2)
```

## Performance Optimization

### Audio Processing
- Default sample rate: 22,050 Hz (balance of quality and speed)
- Adjustable number of calculation points for Doppler curves
- Velocity smoothing with Savitzky-Golay filter

### Job Management
- In-memory job store for development
- For production, use Redis or database
- Automatic cleanup of old files (24-hour default)

### Recommended Settings
- 2-4 Gunicorn workers for CPU-bound tasks
- Nginx for serving static files
- Consider Celery for true async processing at scale

## Troubleshooting

### Audio File Not Found
**Problem:** Server can't find vehicle audio files

**Solution:**
```bash
# Check files exist
ls -la static/
# Should show: horn.mp3, train.mp3, drone.mp3

# If missing, backend will generate synthetic audio automatically
```

### librosa Import Error
**Problem:** `ImportError: No module named 'librosa'`

**Solution:**
```bash
# Install with all dependencies
pip install librosa soundfile audioread

# On Ubuntu/Debian, may need:
sudo apt-get install libsndfile1 ffmpeg
```

### CORS Errors
**Problem:** Browser blocks API requests from React Native app

**Solution:**
```python
# In app.py, update CORS configuration:
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "https://yourdomain.com"],
        "methods": ["GET", "POST", "OPTIONS"]
    }
})
```

### Memory Issues with Long Durations
**Problem:** Server runs out of memory for long audio

**Solution:**
```python
# Limit maximum duration in app.py:
MAX_DURATION = 30  # seconds
audio_duration = min(float(data.get('audio_duration', 5)), MAX_DURATION)
```

## API Rate Limiting

For production, implement rate limiting:

```bash
pip install Flask-Limiter
```

```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.remote_addr,
    default_limits=["100 per hour", "20 per minute"]
)

@app.route('/api/simulate', methods=['POST'])
@limiter.limit("5 per minute")
def api_simulate():
    # ... simulation code
```

## Research Features

### Custom Acceleration Profiles
Enable realistic acceleration/deceleration patterns:

```json
{
  "acceleration_mode": "custom",
  "custom_accel_params": {
    "max_accel": 3.0,
    "max_decel": 5.0,
    "frequency": 0.2
  }
}
```

### Manual Speed Profiles
Define exact speed at each time point:

```json
{
  "acceleration_mode": "manual",
  "manual_profile": {
    "time_values": "0,2,4,6,8,10",
    "speed_values": "0,25,10,30,5,20"
  }
}
```

## Contributing

Contributions welcome! Areas for improvement:
- Additional vehicle types (motorcycle, helicopter, etc.)
- 3D trajectory support
- Real-time streaming audio
- Machine learning-based sound synthesis
- Advanced atmospheric effects (temperature, wind)

## License

This project is part of research at Carnegie Mellon University's Language Technologies Institute under Professor Bhiksha Raj.

## Author

**Rohith**  
Computer Science and Engineering  
Sri Sivasubramaniya Nadar College of Engineering  
Research Intern, Carnegie Mellon University LTI

## Acknowledgments

- Carnegie Mellon University, Language Technologies Institute
- Professor Bhiksha Raj for research guidance
- Acoustic physics based on classical Doppler effect theory
- Audio processing built on librosa and scipy libraries

## Citation

If you use this simulator in academic research, please cite:

```bibtex
@software{doppler_simulator,
  author = {Rohith},
  title = {Doppler Effect Simulator: Acoustic Analysis via Doppler Effects},
  year = {2025},
  institution = {Carnegie Mellon University},
  type = {Research Software}
}
```

---

**Technical Support**: For issues or questions, please open an issue on the GitHub repository.

**API Status**: Monitor backend health at `https://doppler-simulator.duckdns.org/health`
