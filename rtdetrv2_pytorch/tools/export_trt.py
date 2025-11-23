import os
import argparse
import tensorrt as trt

def main(onnx_path, engine_path, max_batchsize, opt_batchsize, min_batchsize, use_fp16=True, verbose=False)->None:
    """ Convert ONNX model to TensorRT engine using TensorRT 10.x APIs.
    Args:
        onnx_path (str): Path to the input ONNX model.
        engine_path (str): Path to save the output TensorRT engine.
        max_batchsize (int): Maximum batch size.
        opt_batchsize (int): Optimal batch size.
        min_batchsize (int): Minimum batch size.
        use_fp16 (bool): Whether to use FP16 precision.
        verbose (bool): Whether to enable verbose logging.
    """
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    
    print(f"[INFO] Loading ONNX file from {onnx_path}")
    
    # Create builder and network
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    # Parse ONNX model
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("[ERROR] Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(f"  - {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX file")

    print(f"[INFO] Successfully parsed ONNX model")
    
    # Create builder config (TensorRT 10.x API)
    config = builder.create_builder_config()
    
    # Set memory pool limit (replaces max_workspace_size in TensorRT 10.x)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # Enable FP16 if supported and requested
    if use_fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] FP16 optimization enabled")
        else:
            print("[WARNING] FP16 not supported on this platform, using FP32")

    # Create optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "images", 
        min=(min_batchsize, 3, 640, 640), 
        opt=(opt_batchsize, 3, 640, 640), 
        max=(max_batchsize, 3, 640, 640)
    )
    profile.set_shape(
        "orig_target_sizes", 
        min=(min_batchsize, 2), 
        opt=(opt_batchsize, 2), 
        max=(max_batchsize, 2)
    )
    config.add_optimization_profile(profile)

    print("[INFO] Building TensorRT engine (this may take several minutes)...")
    
    # Build serialized network (TensorRT 10.x API)
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        raise RuntimeError("Failed to build the TensorRT engine")

    print(f"[INFO] Saving engine to {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f"[INFO] Engine export complete! Saved to {engine_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT Engine")
    parser.add_argument("--onnx", "-i", type=str, required=True, help="Path to input ONNX model file")
    parser.add_argument("--saveEngine", "-o", type=str, default="model.engine", help="Path to output TensorRT engine file")
    parser.add_argument("--maxBatchSize", "-Mb", type=int, default=4, help="Maximum batch size for inference")
    parser.add_argument("--optBatchSize", "-ob", type=int, default=4, help="Optimal batch size for inference")
    parser.add_argument("--minBatchSize", "-mb", type=int, default=4, help="Minimum batch size for inference")
    parser.add_argument("--fp16", default=True, action="store_true", help="Enable FP16 precision mode")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable FP16 precision mode (use FP32)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    main(
        onnx_path=args.onnx,
        engine_path=args.saveEngine,
        max_batchsize=args.maxBatchSize,
        opt_batchsize=args.optBatchSize,
        min_batchsize=args.minBatchSize,
        use_fp16=args.fp16,
        verbose=args.verbose
    )
