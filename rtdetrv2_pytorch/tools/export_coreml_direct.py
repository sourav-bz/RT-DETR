"""
Convert PyTorch RT-DETR model directly to CoreML format (RECOMMENDED METHOD)
This is the modern, officially recommended approach by Apple for PyTorch to CoreML conversion.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import torch
import coremltools as ct
from src.core import YAMLConfig


def convert_pytorch_to_coreml(
    config_path,
    checkpoint_path,
    output_path,
    input_size=640,
    compute_precision='float32',
    minimum_deployment_target='iOS15'
):
    """
    Convert PyTorch RT-DETR model directly to CoreML format
    
    Args:
        config_path: Path to model config file
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Path to save CoreML model
        input_size: Input image size
        compute_precision: Compute precision ('float32' or 'float16')
        minimum_deployment_target: Minimum iOS version
    """
    print(f"\n{'='*80}")
    print(f"🔧 Converting PyTorch RT-DETR to CoreML (Direct Method)")
    print(f"{'='*80}")
    print(f"  Config: {config_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output: {output_path}")
    print(f"  Input size: {input_size}")
    print(f"  Compute precision: {compute_precision}")
    print(f"  Minimum deployment: {minimum_deployment_target}")
    
    # Map deployment target strings to coremltools enums
    deployment_targets = {
        'iOS13': ct.target.iOS13,
        'iOS14': ct.target.iOS14,
        'iOS15': ct.target.iOS15,
        'iOS16': ct.target.iOS16,
        'iOS17': ct.target.iOS17,
        'iOS18': ct.target.iOS18,
    }
    
    target = deployment_targets.get(minimum_deployment_target, ct.target.iOS15)
    
    # Load PyTorch model
    print("\n📦 Loading PyTorch model...")
    
    # Load config
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    
    # Disable pretrained backbone loading
    for backbone_type in ['PResNet', 'HGNetv2']:
        if backbone_type in cfg.yaml_cfg:
            cfg.yaml_cfg[backbone_type]['pretrained'] = False
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state dict
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    elif 'model' in checkpoint:
        state = checkpoint['model']
    else:
        raise ValueError("Checkpoint must contain 'ema' or 'model' key")
    
    # Load state dict
    cfg.model.load_state_dict(state)
    
    # Create deployment model (this removes training-specific components)
    model = cfg.model.deploy()
    postprocessor = cfg.postprocessor.deploy()
    
    # Wrap model and postprocessor
    class RTDETRWrapper(torch.nn.Module):
        def __init__(self, model, postprocessor):
            super().__init__()
            self.model = model
            self.postprocessor = postprocessor
        
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    wrapped_model = RTDETRWrapper(model, postprocessor)
    wrapped_model.eval()
    
    print("✅ PyTorch model loaded successfully!")
    
    # Create example inputs for tracing
    print(f"\n🔍 Tracing model with input size {input_size}x{input_size}...")
    example_image = torch.rand(1, 3, input_size, input_size)
    example_size = torch.tensor([[input_size, input_size]], dtype=torch.int64)
    
    # Trace the model
    try:
        traced_model = torch.jit.trace(wrapped_model, (example_image, example_size))
        print("✅ Model traced successfully!")
    except Exception as e:
        print(f"❌ Error tracing model: {e}")
        print("\nTrying with torch.jit.script instead...")
        try:
            traced_model = torch.jit.script(wrapped_model)
            print("✅ Model scripted successfully!")
        except Exception as e2:
            print(f"❌ Error scripting model: {e2}")
            raise
    
    # Convert to CoreML
    print(f"\n🔄 Converting to CoreML (this may take several minutes)...")
    
    try:
        # Set compute precision
        if compute_precision == 'float16':
            compute_precision_enum = ct.precision.FLOAT16
        else:
            compute_precision_enum = ct.precision.FLOAT32
        
        # Define inputs
        # Note: CoreML's ImageType automatically handles normalization
        inputs = [
            ct.TensorType(
                name="images",
                shape=(1, 3, input_size, input_size),
                dtype=float
            ),
            ct.TensorType(
                name="orig_target_sizes",
                shape=(1, 2),
                dtype=int
            )
        ]
        
        # Convert
        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            minimum_deployment_target=target,
            compute_precision=compute_precision_enum,
            convert_to="mlprogram",  # Modern ML Program format
        )
        
        print("✅ Conversion successful!")
        
        # Add model metadata
        mlmodel.author = "RT-DETR"
        mlmodel.license = "Apache 2.0"
        mlmodel.short_description = f"RT-DETR object detection model ({input_size}x{input_size})"
        mlmodel.version = "1.0"
        
        # Add input descriptions
        mlmodel.input_description['images'] = f"Input image tensor (RGB, {input_size}x{input_size})"
        mlmodel.input_description['orig_target_sizes'] = "Original image size [width, height]"
        
        # Save the model
        print(f"\n💾 Saving CoreML model to: {output_path}")
        mlmodel.save(output_path)
        
        # Print model info
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"\n{'='*80}")
        print(f"✅ CoreML Export Complete!")
        print(f"{'='*80}")
        print(f"\n📊 Model Information:")
        print(f"  File size: {file_size:.2f} MB")
        print(f"  Format: ML Program (.mlpackage)")
        print(f"  Precision: {compute_precision}")
        print(f"  Minimum deployment: {minimum_deployment_target}")
        print(f"  Input size: {input_size}x{input_size}")
        
        print(f"\n📱 Usage in iOS:")
        print(f"  1. Add {os.path.basename(output_path)} to your Xcode project")
        print(f"  2. Xcode will auto-generate Swift wrapper classes")
        print(f"  3. Use MLMultiArray or CVPixelBuffer for input")
        
        return mlmodel
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch RT-DETR directly to CoreML (Recommended Method)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion (float32, iOS 15+)
  python export_coreml_direct.py \\
    -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \\
    -r /path/to/checkpoint.pth \\
    -o rtdetr.mlpackage
  
  # With float16 precision (smaller, faster on modern devices)
  python export_coreml_direct.py \\
    -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \\
    -r /path/to/checkpoint.pth \\
    -o rtdetr_fp16.mlpackage \\
    --precision float16
  
  # Custom input size
  python export_coreml_direct.py \\
    -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \\
    -r /path/to/checkpoint.pth \\
    -o rtdetr_512.mlpackage \\
    --input-size 512
  
  # For older iOS versions
  python export_coreml_direct.py \\
    -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \\
    -r /path/to/checkpoint.pth \\
    -o rtdetr_ios13.mlpackage \\
    --deployment iOS13

Notes:
  - This is the RECOMMENDED method (PyTorch → CoreML directly)
  - Produces modern ML Program format (.mlpackage)
  - Better optimization than ONNX → CoreML path
  - Supports iOS 13+ (iOS 15+ recommended for best features)
  - Float16 reduces model size by ~50% with minimal accuracy loss
        """
    )
    
    # Required arguments
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to model config file (.yml)')
    parser.add_argument('--checkpoint', '-r', type=str, required=True,
                       help='Path to PyTorch checkpoint (.pth)')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Path to output CoreML model (.mlpackage)')
    
    # Model configuration
    parser.add_argument('--input-size', '-s', type=int, default=640,
                       help='Input image size (default: 640)')
    
    # Optimization options
    parser.add_argument('--precision', '-p', type=str, default='float32',
                       choices=['float32', 'float16'],
                       help='Compute precision (default: float32)')
    
    # Deployment options
    parser.add_argument('--deployment', '-d', type=str, default='iOS15',
                       choices=['iOS13', 'iOS14', 'iOS15', 'iOS16', 'iOS17', 'iOS18'],
                       help='Minimum iOS deployment target (default: iOS15)')
    
    args = parser.parse_args()
    
    # Ensure output has correct extension
    if not args.output.endswith('.mlpackage'):
        args.output = args.output.replace('.mlmodel', '.mlpackage')
        if not args.output.endswith('.mlpackage'):
            args.output = args.output + '.mlpackage'
    
    # Convert
    try:
        convert_pytorch_to_coreml(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            input_size=args.input_size,
            compute_precision=args.precision,
            minimum_deployment_target=args.deployment
        )
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
