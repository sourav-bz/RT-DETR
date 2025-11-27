"""
Convert ONNX RT-DETR model to CoreML format for iOS on-device inference
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import numpy as np
import coremltools as ct
from coremltools.models.neural_network import quantization_utils


def convert_onnx_to_coreml(
    onnx_path,
    output_path,
    input_size=640,
    quantize=False,
    compute_precision='float32',
    minimum_deployment_target='iOS15'
):
    """
    Convert ONNX model to CoreML format
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save CoreML model
        input_size: Input image size
        quantize: Whether to quantize the model (reduce size)
        compute_precision: Compute precision ('float32', 'float16', or 'mixed')
        minimum_deployment_target: Minimum iOS version ('iOS13', 'iOS14', 'iOS15', 'iOS16', 'iOS17')
    """
    print(f"\n{'='*80}")
    print(f"🔧 Converting ONNX to CoreML")
    print(f"{'='*80}")
    print(f"  ONNX model: {onnx_path}")
    print(f"  Output path: {output_path}")
    print(f"  Input size: {input_size}")
    print(f"  Compute precision: {compute_precision}")
    print(f"  Minimum deployment: {minimum_deployment_target}")
    
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    # Map deployment target strings to coremltools enums
    deployment_targets = {
        'iOS13': ct.target.iOS13,
        'iOS14': ct.target.iOS14,
        'iOS15': ct.target.iOS15,
        'iOS16': ct.target.iOS16,
        'iOS17': ct.target.iOS17,
    }
    
    target = deployment_targets.get(minimum_deployment_target, ct.target.iOS15)
    
    # Define input shape with flexible batch size
    # Shape: (batch, channels, height, width)
    image_input = ct.ImageType(
        name="images",
        shape=(1, 3, input_size, input_size),
        scale=1.0/255.0,  # Normalize to [0, 1]
        bias=[0, 0, 0],
        color_layout=ct.colorlayout.RGB
    )
    
    # Define orig_target_sizes input
    # Shape: (batch, 2) - [width, height]
    size_input = ct.TensorType(
        name="orig_target_sizes",
        shape=(1, 2),
        dtype=np.int64
    )
    
    print("\n📦 Loading ONNX model...")
    
    # Convert ONNX to CoreML
    try:
        # Set compute precision
        if compute_precision == 'float16':
            compute_precision_enum = ct.precision.FLOAT16
        elif compute_precision == 'mixed':
            # Mixed precision: use float16 where possible, float32 for critical ops
            compute_precision_enum = ct.precision.FLOAT16
        else:
            compute_precision_enum = ct.precision.FLOAT32
        
        print(f"\n🔄 Converting to CoreML (this may take a few minutes)...")
        
        # Use the ONNX-specific converter
        mlmodel = ct.converters.onnx.convert(
            model=onnx_path,
            minimum_ios_deployment_target='13',  # iOS 13+ for better operator support
        )
        
        print("✅ Conversion successful!")
        
        # Add model metadata
        mlmodel.author = "RT-DETR"
        mlmodel.license = "Apache 2.0"
        mlmodel.short_description = "RT-DETR object detection model"
        mlmodel.version = "1.0"
        
        # Add input descriptions
        mlmodel.input_description['images'] = f"Input image (RGB, {input_size}x{input_size})"
        mlmodel.input_description['orig_target_sizes'] = "Original image size [width, height]"
        
        # Add output descriptions
        mlmodel.output_description['labels'] = "Predicted class labels"
        mlmodel.output_description['boxes'] = "Predicted bounding boxes [x1, y1, x2, y2]"
        mlmodel.output_description['scores'] = "Confidence scores"
        
        # Quantize if requested
        if quantize:
            print("\n⚙️  Quantizing model to reduce size...")
            mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=8)
            print("✅ Quantization complete!")
        
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
        print(f"  Format: ML Program")
        print(f"  Precision: {compute_precision}")
        print(f"  Quantized: {'Yes (8-bit)' if quantize else 'No'}")
        print(f"  Minimum deployment: {minimum_deployment_target}")
        
        print(f"\n📱 Usage in iOS:")
        print(f"  1. Add {os.path.basename(output_path)} to your Xcode project")
        print(f"  2. Import CoreML: import CoreML")
        print(f"  3. Load model: let model = try {os.path.splitext(os.path.basename(output_path))[0]}()")
        print(f"  4. Run inference with CVPixelBuffer input")
        
        return mlmodel
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Convert ONNX RT-DETR model to CoreML for iOS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python export_coreml.py -i model.onnx -o model.mlpackage
  
  # With float16 precision (smaller, faster on modern devices)
  python export_coreml.py -i model.onnx -o model.mlpackage --precision float16
  
  # With quantization (even smaller)
  python export_coreml.py -i model.onnx -o model.mlpackage --quantize
  
  # For older iOS versions
  python export_coreml.py -i model.onnx -o model.mlpackage --deployment iOS13
  
  # Full example with all options
  python export_coreml.py -i model.onnx -o model.mlpackage --input-size 640 --precision float16 --quantize --deployment iOS15

Notes:
  - Requires: pip install coremltools onnx
  - ML Program format (.mlpackage) requires iOS 15+ (recommended)
  - Float16 precision reduces model size by ~50% with minimal accuracy loss
  - Quantization further reduces size but may impact accuracy
  - Test on actual devices to verify performance
        """
    )
    
    # Required arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input ONNX model file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Path to output CoreML model file (.mlpackage)')
    
    # Model configuration
    parser.add_argument('--input-size', '-s', type=int, default=640,
                       help='Input image size (default: 640)')
    
    # Optimization options
    parser.add_argument('--precision', '-p', type=str, default='float32',
                       choices=['float32', 'float16', 'mixed'],
                       help='Compute precision (default: float32)')
    parser.add_argument('--quantize', '-q', action='store_true',
                       help='Quantize weights to 8-bit (reduces model size)')
    
    # Deployment options
    parser.add_argument('--deployment', '-d', type=str, default='iOS15',
                       choices=['iOS13', 'iOS14', 'iOS15', 'iOS16', 'iOS17'],
                       help='Minimum iOS deployment target (default: iOS15)')
    
    args = parser.parse_args()
    
    # Ensure output has correct extension
    if not args.output.endswith('.mlpackage') and not args.output.endswith('.mlmodel'):
        args.output = args.output + '.mlpackage'
    
    # Convert
    try:
        convert_onnx_to_coreml(
            onnx_path=args.input,
            output_path=args.output,
            input_size=args.input_size,
            quantize=args.quantize,
            compute_precision=args.precision,
            minimum_deployment_target=args.deployment
        )
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
