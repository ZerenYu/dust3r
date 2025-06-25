# Dust3r Inference Service

This directory contains the gRPC inference service for Dust3r depth prediction.

## Files

- `service.py` - Main service implementation
- `start_server.py` - Script to start the gRPC server
- `test_with_base_name.py` - Test client using real images from data folder
- `README.md` - This file

## Quick Start

### 1. Start the Server

First, start the gRPC server:

```bash
# Start with default settings (dust3r-outdoor model, port 50051)
python start_server.py

# Or with custom settings
python start_server.py --port 50052 --model-name dust3r-indoor --device cpu
```

### 2. Test the Service

In a separate terminal, run the test client:

```bash
# List available image pairs in data folder
python test_with_base_name.py --list-available

# Test with default base name (20250528114015)
python test_with_base_name.py

# Test with specific base name
python test_with_base_name.py --base-name 20250528114015

# Test with custom data directory
python test_with_base_name.py --base-name 20250528114015 --data-dir /path/to/data
```

## Server Options

The `start_server.py` script supports the following options:

- `--port`: Port to run the server on (default: 50051)
- `--model-name`: Model name to use (default: dust3r-outdoor)
- `--model-path`: Path to custom model weights (optional)
- `--device`: Device to run inference on (default: cuda)
- `--max-workers`: Maximum number of worker threads (default: 10)

## Test Client Options

The `test_with_base_name.py` script supports the following options:

- `--base-name`: Base name for camera IDs (default: 20250528114015)
- `--server`: Server address (default: localhost:50051)
- `--data-dir`: Data directory containing images (default: data)
- `--list-available`: List all available base names in the data directory

## Available Models

- `dust3r-outdoor`: Pre-trained model for outdoor scenes
- `dust3r-indoor`: Pre-trained model for indoor scenes

## Data Format

The test client expects stereo image pairs in the data folder with the following naming convention:
- `{base_name}_left.png` - Left camera image
- `{base_name}_right.png` - Right camera image

Example:
```
data/
├── 20250528114015_left.png
├── 20250528114015_right.png
├── 20250528114026_left.png
├── 20250528114026_right.png
└── ...
```

## Testing

### Test with Real Images

The `test_with_base_name.py` script loads real stereo images from the data folder and creates realistic camera parameters:

- **Images**: Loads actual left/right stereo pairs
- **Camera IDs**: Uses base name with `_left` and `_right` suffixes
- **Intrinsics**: Realistic focal length (1200px) and principal point (320, 240)
- **Extrinsics**: Typical stereo baseline (0.12m separation)
- **Distortion**: Slight lens distortion for realism

### Expected Output

The test script will:
1. Connect to the server
2. Load real images from the data folder
3. Send a request with stereo camera images
4. Receive depth and confidence maps
5. Save visualization results as PNG files
6. Display statistics about the results

## Service API

The service implements the `Dust3rInferenceService` gRPC service with the following method:

### PredictDepth

**Request:**
- `captured_images`: List of camera images with metadata
- `reference_camera_id`: ID of the reference camera
- `camera_info`: Camera rig calibration parameters

**Response:**
- `depth_map`: Predicted depth map for the reference camera
- `confidence_map`: Confidence scores for the depth predictions

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the correct directory and the dust3r package is in your Python path.

2. **CUDA Errors**: If you don't have a GPU, use `--device cpu` when starting the server.

3. **Port Already in Use**: Change the port using `--port` option.

4. **Model Loading Errors**: Make sure the model name is correct or provide a valid model path.

5. **Image Not Found**: Check that the base name exists in the data folder and has both left and right images.

### Debug Mode

To see more detailed error information, you can modify the test scripts to include more verbose logging.

## Dependencies

Make sure you have the following dependencies installed:
- torch
- grpcio
- grpcio-tools
- numpy
- PIL
- matplotlib (for visualization)

## Example Usage

```bash
# List all available image pairs
python test_with_base_name.py --list-available

# Test with a specific timestamp
python test_with_base_name.py --base-name 20250528114015

# Test with custom server and data directory
python test_with_base_name.py --base-name 20250528114015 --server localhost:50052 --data-dir /path/to/data
```

```python
import asyncio
import grpc
from dust3r.proto.inference import predict_depth_with_dust3r_pb2_grpc

async def test_service():
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = predict_depth_with_dust3r_pb2_grpc.Dust3rInferenceServiceStub(channel)
        # Create your request here
        response = await stub.PredictDepth(request)
        # Process response

asyncio.run(test_service())
``` 