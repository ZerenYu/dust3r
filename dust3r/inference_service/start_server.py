#!/usr/bin/env python3
"""
Start the Dust3r Inference Service
This script provides a simple way to start the gRPC server
"""

import asyncio
import argparse
import sys
import os

# Add the dust3r package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dust3r.inference_service.service import serve

async def main():
    """Main function to start the server."""
    parser = argparse.ArgumentParser(description='Start Dust3r Inference Service')
    parser.add_argument('--port', type=int, default=50062, help='Port to run the server on (default: 50062)')
    parser.add_argument('--model-name', type=str, default='dust3r-outdoor', 
                       help='Model name to use (default: dust3r-outdoor)')
    parser.add_argument('--model-path', type=str, default='checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth',
                       help='Path to custom model weights (optional)')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to run inference on (default: cuda)')
    parser.add_argument('--max-workers', type=int, default=10,
                       help='Maximum number of worker threads (default: 10)')
    
    args = parser.parse_args()
    
    print("Starting Dust3r Inference Service...")
    print(f"Port: {args.port}")
    print(f"Model: {args.model_name if args.model_path is None else args.model_path}")
    print(f"Device: {args.device}")
    print(f"Max workers: {args.max_workers}")
    
    try:
        # Start the server
        server = serve(
            model_path=args.model_path,
            model_name=args.model_name,
            port=args.port,
            max_workers=args.max_workers
        )
        
        await server.start()
        print(f"Server started successfully on port {args.port}")
        print("Press Ctrl+C to stop the server")
        
        # Wait for termination
        await server.wait_for_termination()
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
        await server.stop(0)
        print("Server stopped")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 