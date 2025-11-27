"""
Convert PyTorch RT-DETR model to CoreML with dtype fixes
This version adds explicit dtype casting to fix gather operation issues
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import torch
import torch.nn as nn
import coremltools as ct
from src.core import YAMLConfig


class RTDETRWrapperWithDtypeFix(nn.Module):
    """
    Wrapper that fixes dtype issues for CoreML conversion
    """
    def __init__(self, model, postprocessor):
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor
    
    def forward(self, images, orig_target_sizes):
        # Run model
        outputs = self.model(images)
        
        # Manually apply postprocessing with dtype fixes
        # Extract outputs
        logits = outputs['pred_logits']  # [B, num_queries, num_classes]
        boxes = outputs['pred_boxes']    # [B, num_queries, 4]
        
        # Apply sigmoid to get scores
        scores = torch.sigmoid(logits)
        
        # Get max score and class per query
        scores_max, labels = torch.max(scores, dim=-1)  # [B, num_queries]
        
        # Convert boxes from cxcywh to xyxy format
        # boxes are in normalized coordinates [cx, cy, w, h]
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
        
        # Scale boxes to original image size
        # orig_target_sizes is [B, 2] where columns are [width, height]
        img_w = orig_target_sizes[:, 0:1]  # [B, 1]
        img_h = orig_target_sizes[:, 1:2]  # [B, 1]
        scale = torch.cat([img_w, img_h, img_w, img_h], dim=-1).unsqueeze(1)  # [B, 1, 4]
        boxes_scaled = boxes_xyxy * scale
        
        # Apply confidence threshold (0.5) and get top-k
        threshold = 0.3
        mask = scores_max > threshold
        
        # For simplicity, return all detections (filtering can be done in iOS)
        # Return: labels (int64), boxes (float32), scores (float32)
        return labels.long(), boxes_scaled, scores_max


def convert_pytorch_to_coreml_with_fix(
    config_path,
    checkpoint_path,
    output_path,
    input_size=640,
    compute_precision='float32',
    minimum_deployment_target='iOS15'
):
    """
    Convert PyTorch RT-DETR model to CoreML with dtype fixes
    """
    print(f"\n{'='*80}")
    print(f"🔧 Converting PyTorch RT-DETR to CoreML (With Dtype Fixes)")
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
    
    # Create deployment model
    model = cfg.model.deploy()
    postprocessor = cfg.postprocessor.deploy()
    
    # Wrap with dtype fix
    wrapped_model = RTDETRWrapperWithDtypeFix(model, postprocessor)
    wrapped_model.eval()
    
    print("✅ PyTorch model loaded successfully!")
    
    # Create example inputs for tracing
    print(f"\n🔍 Tracing model with input size {input_size}x{input_size}...")
    example_image = torch.rand(1, 3, input_size, input_size)
    example_size = torch.tensor([[input_size, input_size]], dtype=torch.int64)
    
    # Trace the model
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(wrapped_model, (example_image, example_size))
        print("✅ Model traced successfully!")
    except Exception as e:
        print(f"❌ Error tracing model: {e}")
        import traceback
        traceback.print_exc()
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
        
        print(f"\n📋 Model Outputs:")
        print(f"  1. labels: Class IDs (int64) - shape [1, 300]")
        print(f"  2. boxes: Bounding boxes in [x1, y1, x2, y2] format (float32) - shape [1, 300, 4]")
        print(f"  3. scores: Confidence scores (float32) - shape [1, 300]")
        
        print(f"\n⚠️  Note:")
        print(f"  - Outputs include ALL 300 queries")
        print(f"  - You still need to filter by score threshold in iOS")
        print(f"  - You may want to apply NMS to remove duplicates")
        print(f"  - Default threshold in model: 0.3 (modify in code if needed)")
        
        print(f"\n📱 Usage in iOS:")
        print(f"  1. Add {os.path.basename(output_path)} to your Xcode project")
        print(f"  2. Xcode will auto-generate Swift wrapper classes")
        print(f"  3. Filter detections by score > threshold")
        print(f"  4. Optionally apply NMS")
        
        return mlmodel
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch RT-DETR to CoreML (With Dtype Fixes)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion (float32, iOS 15+)
  python export_coreml_fixed.py \\
    -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \\
    -r /path/to/checkpoint.pth \\
    -o rtdetr_fixed.mlpackage
  
  # With float16 precision (smaller, faster)
  python export_coreml_fixed.py \\
    -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \\
    -r /path/to/checkpoint.pth \\
    -o rtdetr_fixed_fp16.mlpackage \\
    --precision float16

Notes:
  - This version includes custom postprocessing with dtype fixes
  - Outputs: labels (int64), boxes (float32), scores (float32)
  - All 300 detections included - filter by score in iOS
  - Boxes in [x1, y1, x2, y2] format scaled to original image size
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
        convert_pytorch_to_coreml_with_fix(
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
