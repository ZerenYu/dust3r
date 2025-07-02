import os
import torch
import numpy as np
from PIL import Image
import grpc
import asyncio
from concurrent import futures

from dust3r.model import load_model, AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.image import ImgNorm, rgb

# Import the generated protobuf code
from dust3r.proto.inference import predict_depth_with_dust3r_pb2 as dust3r_pb2
from dust3r.proto.inference import predict_depth_with_dust3r_pb2_grpc as dust3r_pb2_grpc

class Dust3rInferenceServicer(dust3r_pb2_grpc.Dust3rInferenceServiceServicer):
    def __init__(self, model_path=None, model_name=None, device='cuda'):
        """Initialize the servicer with either a model path or model name."""
        self.device = device
        if model_path:
            self.model = load_model(model_path, device)
        elif model_name:
            self.model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
        else:
            raise ValueError("Either model_path or model_name must be provided")
        
        self.model.eval()

    def _create_transformation_matrix(self, position, orientation_quat):
        """Creates a 4x4 transformation matrix from position and quaternion."""
        pos = torch.tensor([position.x, position.y, position.z], device=self.device)
        quat = torch.tensor([orientation_quat.w, orientation_quat.x, orientation_quat.y, orientation_quat.z], device=self.device) # w, x, y, z

        # Normalize quaternion
        quat = quat / torch.linalg.norm(quat)

        # Convert quaternion to rotation matrix
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        rotation_matrix = torch.eye(3, device=self.device)
        rotation_matrix[0, 0] = 1 - 2*y*y - 2*z*z
        rotation_matrix[0, 1] = 2*x*y - 2*z*w
        rotation_matrix[0, 2] = 2*x*z + 2*y*w
        rotation_matrix[1, 0] = 2*x*y + 2*z*w
        rotation_matrix[1, 1] = 1 - 2*x*x - 2*z*z
        rotation_matrix[1, 2] = 2*y*z - 2*x*w
        rotation_matrix[2, 0] = 2*x*z - 2*y*w
        rotation_matrix[2, 1] = 2*y*z + 2*x*w
        rotation_matrix[2, 2] = 1 - 2*x*x - 2*y*y

        # Create transformation matrix
        transform = torch.eye(4, device=self.device)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = pos
        return transform

    def _prepare_image_for_dust3r(self, camera_image, idx, size=512):
        """Convert proto CameraImage to Dust3r format, with resizing."""
        # Convert proto image data to PIL Image
        img_data = np.frombuffer(camera_image.image.raw_data, dtype=np.uint8)
        img_data = img_data.reshape(
            camera_image.image.metadata.height,
            camera_image.image.metadata.width,
            3  # Assuming RGB
        )
        
        pil_img = Image.fromarray(img_data)
        
        # Resize logic from dust3r/utils/image.py
        W1, H1 = pil_img.size
        
        # resize long side to 'size'
        S = max(H1, W1)
        if S > size:
            interp = Image.LANCZOS
        else:
            interp = Image.BICUBIC
        
        new_size = tuple(int(round(x * size / S)) for x in pil_img.size)
        resized_img = pil_img.resize(new_size, interp)

        W, H = resized_img.size
        cx, cy = W // 2, H // 2
        halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
        cropped_img = resized_img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
        print(f"Cropped image size: {cropped_img.size}")
        # Convert to Dust3r format
        img_norm = ImgNorm(cropped_img)
        return {
            'img': img_norm[None],  # Add batch dimension
            'true_shape': np.int32([cropped_img.size[::-1]]),
            'idx': idx,
            'instance': str(idx)
        }

    def _prepare_camera_info(self, camera_info, image_data):
        """Extract camera parameters from proto message."""
        # This will be used for camera calibration in the inference
        focals = []
        for intr in camera_info.intrinsics:
            focals.append((intr.fx, intr.fy))
        return focals

    async def PredictDepth(self, request, context):
        """Implement the PredictDepth RPC method."""
        print("Got request")

        # Convert images to Dust3r format
        loaded_imgs = []
        for idx, cam_img in enumerate(request.captured_images):
            img_data = self._prepare_image_for_dust3r(cam_img, idx)
            loaded_imgs.append(img_data)
        print("Loaded images")
        
        # Save loaded_imgs to output for debugging
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, img_data in enumerate(loaded_imgs):
            img_tensor = img_data['img']
            img_array = rgb(img_tensor)[0]  # remove batch dim
            img_to_save = Image.fromarray((img_array * 255).astype(np.uint8))
            img_to_save.save(os.path.join(output_dir, f"debug_loaded_img_{i}.png"))
        print("Debug images saved to 'output' directory.")

        has_camera_info = True
        known_poses = []
        known_focals = []

        if has_camera_info:
            print("Processing with known camera info")
            intrinsics_map = {intr.camera_id.id: intr.intrinsic for intr in request.camera_info.intrinsics}
            camera_ids = [intr.camera_id.id for intr in request.camera_info.intrinsics]
            extrinsics_map = {cid: ext.transform for cid, ext in zip(camera_ids, request.camera_info.extrinsics)}
            print(f"Intrinsics map: {intrinsics_map}")
            print(f"Extrinsics map: {extrinsics_map}")
            for idx, cam_img in enumerate(request.captured_images):
                camera_id = cam_img.camera_id.id
                img_data = loaded_imgs[idx]

                # Poses
                transform3d = extrinsics_map[camera_id]
                pose = self._create_transformation_matrix(transform3d.pose.position, transform3d.pose.orientation)
                known_poses.append(pose)

                # Focals
                intrinsic = intrinsics_map[camera_id]
                original_h = cam_img.image.metadata.height
                original_w = cam_img.image.metadata.width
                resized_h, resized_w = img_data['img'].shape[2:]

                if resized_h > 0 and resized_w > 0:
                    scale = max(original_w / float(resized_w), original_h / float(resized_h))
                    focal = intrinsic.fx / scale 
                    focal = focal + 200
                    known_focals.append(focal)
            print(f"Original image size: {original_h}x{original_w}")
            print(f"Resized image size: {resized_h}x{resized_w}")
            print(f"Known poses: {known_poses}")
            print(f"Scale: {scale}")
            print(f"Known focals: {known_focals}")
            
            # Replace the second pose with custom rotation and translation
            if len(known_poses) > 1:
                # Custom rotation matrix
                custom_rotation = torch.tensor([
                    [ 0.96875015,  0.0,         0.24803859],
                    [ 0.0,         1.0,        -0.0       ],
                    [-0.24803859, -0.0,         0.96875015]
                ], device=self.device)
                
                # Custom translation vector
                custom_translation = torch.tensor([-0.17423679069, 0.0, 0.02195171761], device=self.device)
                
                # Create custom 4x4 transformation matrix
                custom_transform = torch.eye(4, device=self.device)
                custom_transform[:3, :3] = custom_rotation
                custom_transform[:3, 3] = custom_translation
                # Calculate inverse transformation matrix
                custom_transform_inv = torch.inverse(custom_transform)
                print(f"Inverse custom transformation matrix: {custom_transform_inv}")
                # Replace the second pose
                known_poses[1] = custom_transform
                print(f"Replaced second pose with custom transformation matrix")
                print(f"Custom pose: {custom_transform}")
        
        # Create image pairs
        pairs = make_pairs(loaded_imgs, prefilter=None, symmetrize=True)

        # Run inference
        with torch.no_grad():
            output = inference(pairs, self.model, self.device, batch_size=1)
        
        # Enable gradients for optimization if using known camera info
        torch.autograd.set_grad_enabled(has_camera_info)
        print("after inference")
        # Run global alignment
        if has_camera_info:
            scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.ModularPointCloudOptimizer, optimize_pp=True)
            scene.preset_pose(known_poses, [True] * len(known_poses))
            scene.preset_focal(known_focals, [True] * len(known_focals))
            print("before alignment")
            scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.01)
            print(f"pose from scene: {scene.get_im_poses}")
        else:
            scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PairViewer)
        
        # Disable gradients after optimization
        torch.autograd.set_grad_enabled(False)

        print("Finished inference")
        # Get depth maps and confidence scores
        depth_maps = scene.get_depthmaps()
        confidence_maps = [c for c in scene.im_conf]

        # Find the index of the reference camera
        ref_idx = 0  # Default to first camera if not found
        original_height, original_width = 0, 0
        for idx, cam_img in enumerate(request.captured_images):
            if cam_img.camera_id == request.reference_camera_id:
                ref_idx = idx
                original_height = cam_img.image.metadata.height
                original_width = cam_img.image.metadata.width
                break
        
        # Get the depth map and confidence for the reference camera
        depth_map_tensor = depth_maps[ref_idx]
        conf_map_tensor = confidence_maps[ref_idx]
        # Print depth statistics
        depth_min = depth_map_tensor.min().item()
        depth_max = depth_map_tensor.max().item()
        depth_median = depth_map_tensor.median().item()
        print(f"Depth stats - min: {depth_min:.3f}, max: {depth_max:.3f}, median: {depth_median:.3f}")
        # Rescale to original size
        # Add batch and channel dimensions for interpolate
        rescaled_depth_tensor = torch.nn.functional.interpolate(
            depth_map_tensor.unsqueeze(0).unsqueeze(0),
            size=(original_height, original_width),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        rescaled_conf_tensor = torch.nn.functional.interpolate(
            conf_map_tensor.unsqueeze(0).unsqueeze(0),
            size=(original_height, original_width),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        depth_data = to_numpy(rescaled_depth_tensor)
        conf_data = to_numpy(rescaled_conf_tensor)

        print(f"Depth map dtype: {depth_data.dtype}")
        print(f"Confidence map dtype: {conf_data.dtype}")
        print("Finished conversion")

        # Create response
        response = dust3r_pb2.PredictDust3rDepthResponse()
        
        # Add depth map
        depth_map_out = response.depth_map
        depth_map_out.metadata.height = depth_data.shape[0]
        depth_map_out.metadata.width = depth_data.shape[1]
        depth_map_out.raw_data = depth_data.tobytes()
        
        # Add confidence map
        conf_map_out = response.confidence_map
        conf_map_out.metadata.height = conf_data.shape[0]
        conf_map_out.metadata.width = conf_data.shape[1]
        conf_map_out.raw_data = conf_data.tobytes()

        return response

def serve(model_path=None, model_name=None, port=50051, max_workers=10):
    """Start the gRPC server."""
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    dust3r_pb2_grpc.add_Dust3rInferenceServiceServicer_to_server(
        Dust3rInferenceServicer(model_path, model_name),
        server
    )
    server.add_insecure_port(f'[::]:{port}')
    return server

async def main():
    """Main function to start the server."""
    server = serve(model_name="dust3r-outdoor")  # or use model_path for custom weights
    await server.start()
    print(f"Server started on port 50051")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(main()) 