"""
Convert PyTorch RT-DETR model to CoreML (Simplified - Model Only)
This version exports just the model without postprocessing to avoid dtype issues
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import torch
import coremltools as ct
from src.core import YAMLConfig


def convert_pytorch_to_coreml_simple(
    config_path,
    checkpoint_path,
    output_path,
    input_size=640,
    compute_precision='float32',
    minimum_deployment_target='iOS15'
):
    """
    Convert PyTorch RT-DETR model to CoreML (model only, no postprocessing)
    
    Args:
        config_path: Path to model config file
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Path to save CoreML model
        input_size: Input image size
        compute_precision: Compute precision ('float32' or 'float16')
        minimum_deployment_target: Minimum iOS version
    """
    print(f"\n{'='*80}")
    print(f"🔧 Converting PyTorch RT-DETR to CoreML (Simplified - Model Only)")
    print(f"{'='*80}")
    print(f"  Config: {config_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output: {output_path}")
    print(f"  Input size: {input_size}")
    print(f"  Compute precision: {compute_precision}")
    print(f"  Minimum deployment: {minimum_deployment_target}")
    print(f"\n⚠️  Note: This exports the model WITHOUT postprocessing")
    print(f"  You'll need to implement NMS and score filtering in Swift/ObjC")
    
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
    
    # Create deployment model (WITHOUT postprocessor to avoid dtype issues)
    model = cfg.model.deploy()
    model.eval()
    
    print("✅ PyTorch model loaded successfully!")
    
    # Create example inputs for tracing
    print(f"\n🔍 Tracing model with input size {input_size}x{input_size}...")
    example_image = torch.rand(1, 3, input_size, input_size)
    
    # Trace the model
    try:
        traced_model = torch.jit.trace(model, example_image)
        print("✅ Model traced successfully!")
    except Exception as e:
        print(f"❌ Error tracing model: {e}")
        raise
    
    # Convert to CoreML
    print(f"\n🔄 Converting to CoreML (this may take several minutes)...")
    
    try:
        # Set compute precision
        if compute_precision == 'float16':
            compute_precision_enum = ct.precision.FLOAT16
        else:
            compute_precision_enum = ct.precision.FLOAT32
        
        # Define inputs - just the image
        inputs = [
            ct.TensorType(
                name="images",
                shape=(1, 3, input_size, input_size),
                dtype=float
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
        mlmodel.short_description = f"RT-DETR object detection model ({input_size}x{input_size}) - Raw outputs"
        mlmodel.version = "1.0"
        
        # Add input descriptions
        mlmodel.input_description['images'] = f"Input image tensor (RGB, {input_size}x{input_size})"
        
        # Add output descriptions
        print("\n📝 Model outputs (you need to implement postprocessing):")
        for spec in mlmodel.get_spec().description.output:
            print(f"  - {spec.name}: shape {spec.type.multiArrayType.shape}")
        
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
        
        print(f"\n⚠️  Important:")
        print(f"  This model outputs RAW predictions (logits and boxes)")
        print(f"  You need to implement in Swift/ObjC:")
        print(f"    1. Apply sigmoid to class logits")
        print(f"    2. Convert box coordinates (cxcywh → xyxy)")
        print(f"    3. Apply confidence threshold filtering")
        print(f"    4. Apply NMS (Non-Maximum Suppression)")
        
        print(f"\n📱 Usage in iOS:")
        print(f"  1. Add {os.path.basename(output_path)} to your Xcode project")
        print(f"  2. Xcode will auto-generate Swift wrapper classes")
        print(f"  3. Implement postprocessing logic in Swift")
        
        return mlmodel
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch RT-DETR to CoreML (Simplified - Model Only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion (float32, iOS 15+)
  python export_coreml_simple.py \\
    -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \\
    -r /path/to/checkpoint.pth \\
    -o rtdetr_simple.mlpackage
  
  # With float16 precision (smaller, faster)
  python export_coreml_simple.py \\
    -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \\
    -r /path/to/checkpoint.pth \\
    -o rtdetr_simple_fp16.mlpackage \\
    --precision float16

Notes:
  - This version exports ONLY the model (no postprocessing)
  - Avoids dtype conversion issues
  - You must implement NMS and filtering in Swift/ObjC
  - Outputs are raw logits and bounding boxes
  - Smaller model size than full version
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
        convert_pytorch_to_coreml_simple(
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
