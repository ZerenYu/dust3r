#!/usr/bin/env python3
"""
Test client for Dust3r Inference Service using base name with real images from data folder
This script loads real stereo images and creates test data with realistic camera parameters
"""

import asyncio
import grpc
import numpy as np
from PIL import Image
import os
import sys
import math
import glob

# Add the dust3r package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dust3r.proto.inference import predict_depth_with_dust3r_pb2 as dust3r_pb2
from dust3r.proto.inference import predict_depth_with_dust3r_pb2_grpc as dust3r_pb2_grpc
from dust3r.proto.common import camera_pb2
from dust3r.proto.common import image_pb2
from dust3r.proto.common import ids_pb2
from dust3r.proto.common import calibration_pb2
from dust3r.proto.common import geometry_pb2
from dust3r.proto.common import frame_pb2

def load_and_resize_image(image_path, target_size=(640, 480)):
    """Load and resize an image from the data folder."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(img)

def find_image_pairs(data_dir, base_name):
    """Find left and right image pairs for a given base name."""
    left_pattern = os.path.join(data_dir, f"{base_name}_left.png")
    right_pattern = os.path.join(data_dir, f"{base_name}_right.png")
    
    left_files = glob.glob(left_pattern)
    right_files = glob.glob(right_pattern)
    
    if not left_files:
        raise FileNotFoundError(f"No left image found for base name: {base_name}")
    if not right_files:
        raise FileNotFoundError(f"No right image found for base name: {base_name}")
    
    return left_files[0], right_files[0]

def list_available_base_names(data_dir):
    """List all available base names in the data directory."""
    left_files = glob.glob(os.path.join(data_dir, "*_left.png"))
    base_names = []
    
    for file_path in left_files:
        filename = os.path.basename(file_path)
        base_name = filename.replace("_left.png", "")
        base_names.append(base_name)
    
    return sorted(base_names)

def create_camera_image(img_array, camera_id, fx=1000.0, fy=1000.0, px=320.0, py=240.0):
    """Create a CameraImage proto message from numpy array."""
    camera_image = camera_pb2.CameraImage()
    
    # Set camera ID
    camera_image.camera_id.CopyFrom(ids_pb2.CameraID(id=camera_id))
    
    # Set image data
    camera_image.image.metadata.height = img_array.shape[0]
    camera_image.image.metadata.width = img_array.shape[1]
    camera_image.image.metadata.channel_type = image_pb2.IMAGE_CHANNEL_TYPE_RGB
    camera_image.image.metadata.data_layout = image_pb2.IMAGE_DATA_LAYOUT_HWC
    camera_image.image.metadata.data_type = image_pb2.IMAGE_DATA_TYPE_UINT8
    camera_image.image.metadata.compression = image_pb2.IMAGE_COMPRESSION_RAW
    camera_image.image.raw_data = img_array.tobytes()
    
    # Set intrinsic parameters
    camera_image.intrinsics.fx = fx
    camera_image.intrinsics.fy = fy
    camera_image.intrinsics.px = px
    camera_image.intrinsics.py = py
    camera_image.intrinsics.s = 0.0
    
    # Set distortion parameters (slight distortion for realism)
    camera_image.distortion.five_coeffs_distortion_parameters.k1 = 0.01
    camera_image.distortion.five_coeffs_distortion_parameters.k2 = -0.005
    camera_image.distortion.five_coeffs_distortion_parameters.p1 = 0.001
    camera_image.distortion.five_coeffs_distortion_parameters.p2 = 0.001
    camera_image.distortion.five_coeffs_distortion_parameters.k3 = 0.002
    
    return camera_image

def create_transform_matrix(translation, rotation_degrees):
    """Create a 4x4 transformation matrix from translation and rotation."""
    # Convert rotation from degrees to radians
    rx, ry, rz = [math.radians(r) for r in rotation_degrees]
    
    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)]
    ])
    
    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)]
    ])
    
    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation
    R = Rz @ Ry @ Rx
    
    # Create 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = translation
    
    return transform

def create_test_request_with_base_name(base_name="20250528114015", data_dir="data"):
    """Create a test request using real images from data folder with mock pose and intrinsic."""
    request = dust3r_pb2.PredictDust3rDepthRequest()
    
    # Find image pairs for the base name
    left_image_path, right_image_path = find_image_pairs(data_dir, base_name)
    
    print(f"Loading images for base name: {base_name}")
    print(f"  Left image: {os.path.basename(left_image_path)}")
    print(f"  Right image: {os.path.basename(right_image_path)}")
    
    # Load and resize images
    img_left = load_and_resize_image(left_image_path, (640, 480))
    img_right = load_and_resize_image(right_image_path, (640, 480))
    
    print(f"  Left image shape: {img_left.shape}")
    print(f"  Right image shape: {img_right.shape}")
    
    # Create camera images with realistic parameters
    camera_left = create_camera_image(img_left, f"{base_name}_left", fx=1200.0, fy=1200.0, px=320.0, py=240.0)
    camera_right = create_camera_image(img_right, f"{base_name}_right", fx=1200.0, fy=1200.0, px=320.0, py=240.0)
    
    request.captured_images.extend([camera_left, camera_right])
    
    # Set reference camera (left camera)
    request.reference_camera_id.CopyFrom(ids_pb2.CameraID(id=f"{base_name}_left"))
    
    # Create camera rig calibration with realistic parameters
    camera_rig = calibration_pb2.CameraRigCalibrationParameters()
    camera_rig.camera_rig_id.CopyFrom(ids_pb2.CameraRigID(id=f"{base_name}_rig"))
    
    # Left camera (reference camera) - at origin
    intrinsic_left = calibration_pb2.CameraIntrinsic()
    intrinsic_left.camera_id.CopyFrom(ids_pb2.CameraID(id=f"{base_name}_left"))
    intrinsic_left.intrinsic.fx = 1252.3121337890625
    intrinsic_left.intrinsic.fy = 1333.312255859375
    intrinsic_left.intrinsic.px = 384.77813720703125
    intrinsic_left.intrinsic.py = 385.0728454589844
    intrinsic_left.intrinsic.s = 0.0
    intrinsic_left.height = 480
    intrinsic_left.width = 640
    
    # Right camera - offset to the right (stereo baseline)
    intrinsic_right = calibration_pb2.CameraIntrinsic()
    intrinsic_right.camera_id.CopyFrom(ids_pb2.CameraID(id=f"{base_name}_right"))
    intrinsic_right.intrinsic.fx = 1249.4617919921875
    intrinsic_right.intrinsic.fy = 1315.15380859375
    intrinsic_right.intrinsic.px = 273.49932861328125
    intrinsic_right.intrinsic.py = 393.88031005859375
    intrinsic_right.intrinsic.s = 0.0
    intrinsic_right.height = 480
    intrinsic_right.width = 640
    
    camera_rig.intrinsics.extend([intrinsic_left, intrinsic_right])
    
    # Create extrinsic transformation for right camera
    # Typical stereo baseline: 0.12m to the right, slight forward offset
    transform_right = create_transform_matrix([0.12, 0.0, 0.02], [0, 0, 0])
    extrinsic_right = calibration_pb2.CameraExtrinsic()
    # The Transform3D needs proper structure - this may need to be revised
    # For now, create a simple transform with identity pose
    transform_3d = geometry_pb2.Transform3D()
    transform_3d.pose.position.x = 0.12
    transform_3d.pose.position.y = 0.0
    transform_3d.pose.position.z = 0.02
    transform_3d.pose.orientation.w = 1.0
    transform_3d.pose.orientation.x = 0.0
    transform_3d.pose.orientation.y = 0.0
    transform_3d.pose.orientation.z = 0.0
    extrinsic_right.transform.CopyFrom(transform_3d)
    
    camera_rig.extrinsics.extend([extrinsic_right])
    
    request.camera_info.CopyFrom(camera_rig)
    
    return request

async def test_inference_service(server_address="localhost:50051", base_name="20250528114015", data_dir="data", output_dir="output"):
    """Test the inference service with real images from data folder."""
    print(f"Connecting to server at {server_address}")
    print(f"Using base name: {base_name}")
    print(f"Data directory: {data_dir}")
    
    async with grpc.aio.insecure_channel(server_address) as channel:
        stub = dust3r_pb2_grpc.Dust3rInferenceServiceStub(channel)
        
        # Create test request with real images
        request = create_test_request_with_base_name(base_name, data_dir)
        print(f"Created test request with {len(request.captured_images)} images")
        print(f"Reference camera: {request.reference_camera_id.id}")
        print(f"Camera rig ID: {request.camera_info.camera_rig_id.id}")
        
        # Print camera details
        for i, cam_img in enumerate(request.captured_images):
            print(f"  Camera {i+1}: {cam_img.camera_id.id}")
            print(f"    Image size: {cam_img.image.metadata.width}x{cam_img.image.metadata.height}")
            print(f"    Focal length: ({cam_img.intrinsics.fx:.1f}, {cam_img.intrinsics.fy:.1f})")
            print(f"    Principal point: ({cam_img.intrinsics.px:.1f}, {cam_img.intrinsics.py:.1f})")
        
        try:
            # Call the service
            print("\nSending request to server...")
            response = await stub.PredictDepth(request)
            
            print("Received response from server!")
            print(f"Depth map size: {response.depth_map.metadata.width}x{response.depth_map.metadata.height}")
            print(f"Confidence map size: {response.confidence_map.metadata.width}x{response.confidence_map.metadata.height}")
            
            # Convert depth map back to numpy array
            depth_data = np.frombuffer(response.depth_map.raw_data, dtype=np.float32)
            depth_data = depth_data.reshape(response.depth_map.metadata.height, response.depth_map.metadata.width)
            
            # Convert confidence map back to numpy array
            conf_data = np.frombuffer(response.confidence_map.raw_data, dtype=np.float32)
            conf_data = conf_data.reshape(response.confidence_map.metadata.height, response.confidence_map.metadata.width)
            
            print(f"Depth range: {depth_data.min():.3f} - {depth_data.max():.3f}")
            print(f"Confidence range: {conf_data.min():.3f} - {conf_data.max():.3f}")
            
            # Save results for visualization
            try:
                import matplotlib.pyplot as plt
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot depth map
                im1 = axes[0].imshow(depth_data, cmap='viridis')
                axes[0].set_title(f'Depth Map - {base_name}')
                plt.colorbar(im1, ax=axes[0])
                
                # Plot confidence map
                im2 = axes[1].imshow(conf_data, cmap='plasma')
                axes[1].set_title(f'Confidence Map - {base_name}')
                plt.colorbar(im2, ax=axes[1])
                
                plt.tight_layout()
                
                # Create output directory if it doesn't exist
                if not os.path.exists("output"):
                    os.makedirs("output")
                
                output_path = os.path.join("output", f"{base_name}_result.png")
                plt.savefig(output_path)
                print(f"Saved result to {output_path}")
                
            except ImportError:
                print("Matplotlib not found, skipping visualization.")
                # Save raw data instead
                if not os.path.exists("output"):
                    os.makedirs("output")
                np.save(os.path.join("output", f"{base_name}_depth.npy"), depth_data)
                np.save(os.path.join("output", f"{base_name}_confidence.npy"), conf_data)
                print(f"Saved raw data to output directory")
            
        except grpc.aio.AioRpcError as e:
            print(f"Error calling service: {e.code()} - {e.details()}")

async def main():
    """Main function to test the service."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Dust3r Inference Service with real images')
    parser.add_argument('--base-name', type=str, default="20250528114015",
                        help='Base name for camera IDs (default: 20250528114015)')
    parser.add_argument('--server', type=str, default='localhost:50051',
                        help='Server address (default: localhost:50051)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory containing images (default: data)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save output results (default: output)')
    parser.add_argument('--list-available', action='store_true',
                        help='List all available base names in the data directory')
    
    args = parser.parse_args()
    
    if args.list_available:
        try:
            available_base_names = list_available_base_names(args.data_dir)
            if not available_base_names:
                print(f"No available image pairs found in '{args.data_dir}'")
            else:
                print("Available base names:")
                for name in available_base_names:
                    print(f"  - {name}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
        return
    
    try:
        await test_inference_service(args.server, args.base_name, args.data_dir, args.output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that the data directory and base name are correct.")
        print(f"You can list available base names with --list-available")

if __name__ == "__main__":
    # Add command line arguments for server address, base name, and data directory
    async def run_main():
        await main()
    
    asyncio.run(run_main()) 