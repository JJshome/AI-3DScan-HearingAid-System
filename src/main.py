"""
Main entry point for the AI-Enhanced 3D Scanning Hearing Aid Manufacturing System
"""
import argparse
import logging
import os
import sys
import time
from typing import Dict, Any, List, Optional

from . import __version__
from .hardware import Scanner3DDriver, HighResolutionScanner
from .ai import EarScanModel, PointCloudProcessor
from .config.settings import SYSTEM_CONFIG, SCANNER_CONFIG
from .utils.exceptions import HardwareConnectionError, ScanningError, ProcessingError

# Configure logging
logging.basicConfig(
    level=getattr(logging, SYSTEM_CONFIG.get('log_level', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=SYSTEM_CONFIG.get('log_file', None)
)

logger = logging.getLogger(__name__)


def setup_scanner(device_id: str, high_resolution: bool = False) -> Scanner3DDriver:
    """
    Initialize and connect to the 3D scanner
    
    Args:
        device_id: Scanner device identifier
        high_resolution: Whether to use the high-resolution scanner driver
    
    Returns:
        Scanner3DDriver: Initialized and connected scanner
    """
    logger.info(f"Setting up {'high-resolution ' if high_resolution else ''}scanner {device_id}")
    
    # Create appropriate scanner instance
    scanner_class = HighResolutionScanner if high_resolution else Scanner3DDriver
    scanner = scanner_class(device_id)
    
    # Connect to the scanner
    connected = scanner.connect()
    
    if not connected:
        logger.error(f"Failed to connect to scanner {device_id}")
        raise HardwareConnectionError(f"Could not connect to scanner {device_id}")
    
    logger.info(f"Scanner {device_id} connected successfully")
    return scanner


def perform_scan(scanner: Scanner3DDriver, 
                scan_parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Perform a 3D scan of the ear and process the results
    
    Args:
        scanner: Initialized scanner driver
        scan_parameters: Optional scan parameters
    
    Returns:
        Dict[str, Any]: Scan results including point cloud and metadata
    """
    logger.info("Starting ear scanning process")
    
    # Ensure scanner is calibrated
    scanner.calibrate()
    
    # Perform the actual scan
    start_time = time.time()
    point_cloud = scanner.scan(scan_parameters)
    scan_duration = time.time() - start_time
    
    # Get scanner info
    device_info = scanner.get_device_info()
    
    # Prepare results
    results = {
        "point_cloud": point_cloud,
        "scan_timestamp": time.time(),
        "scan_duration_seconds": scan_duration,
        "num_points": len(point_cloud),
        "device_info": device_info,
        "scan_parameters": scan_parameters or {},
    }
    
    logger.info(f"Scan completed with {len(point_cloud)} points in {scan_duration:.2f} seconds")
    
    return results


def process_scan_data(scan_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the scan data using AI models
    
    Args:
        scan_results: Results from the scanning process
    
    Returns:
        Dict[str, Any]: Processed results including hearing aid model
    """
    logger.info("Processing scan data with AI models")
    
    # Extract point cloud data
    point_cloud = scan_results["point_cloud"]
    
    # Initialize the processor and model
    processor = PointCloudProcessor()
    model = EarScanModel()
    
    # Pre-process the point cloud
    filtered_cloud = processor.filter_outliers(point_cloud)
    
    # Perform alignment
    aligned_cloud, transform = processor.align_to_canonical_orientation(filtered_cloud)
    
    # Compute statistical features
    cloud_features = processor.compute_point_cloud_features(aligned_cloud)
    
    # Process with AI model
    model_results = model.process_scan(aligned_cloud)
    
    # Combine all results
    processing_results = {
        "original_scan": scan_results,
        "preprocessing": {
            "filtered_points": len(filtered_cloud),
            "alignment_transform": transform,
            "statistical_features": cloud_features
        },
        "model_results": model_results,
        "processing_timestamp": time.time(),
        "processing_duration_seconds": model_results["total_processing_time"],
    }
    
    logger.info(f"Scan processing completed in {model_results['total_processing_time']:.2f} seconds")
    
    return processing_results


def run_pipeline(device_id: str, 
               high_resolution: bool = False,
               scan_parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run the complete scanning and processing pipeline
    
    Args:
        device_id: Scanner device identifier
        high_resolution: Whether to use high-resolution scanning
        scan_parameters: Optional scanning parameters
    
    Returns:
        Dict[str, Any]: Complete pipeline results
    """
    logger.info(f"Starting complete pipeline with device {device_id}")
    start_time = time.time()
    
    try:
        # Setup scanner
        scanner = setup_scanner(device_id, high_resolution)
        
        # Perform scan
        scan_results = perform_scan(scanner, scan_parameters)
        
        # Process scan data
        processing_results = process_scan_data(scan_results)
        
        # Disconnect scanner
        scanner.disconnect()
        
        # Add pipeline metadata
        pipeline_duration = time.time() - start_time
        results = {
            "pipeline_duration_seconds": pipeline_duration,
            "pipeline_timestamp": time.time(),
            "pipeline_version": __version__,
            "status": "success",
            "processing_results": processing_results
        }
        
        logger.info(f"Complete pipeline executed successfully in {pipeline_duration:.2f} seconds")
        
        return results
        
    except (HardwareConnectionError, ScanningError, ProcessingError) as e:
        logger.error(f"Pipeline error: {str(e)}")
        return {
            "pipeline_duration_seconds": time.time() - start_time,
            "pipeline_timestamp": time.time(),
            "pipeline_version": __version__,
            "status": "error",
            "error_message": str(e),
            "error_type": e.__class__.__name__
        }


def main():
    """
    Main entry point function when executed as a script
    """
    parser = argparse.ArgumentParser(
        description="AI-Enhanced 3D Scanning Hearing Aid Manufacturing System"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="scanner01",
        help="Scanner device ID to use"
    )
    
    parser.add_argument(
        "--high-res", 
        action="store_true",
        help="Use high-resolution scanning"
    )
    
    parser.add_argument(
        "--scan-mode", 
        type=str, 
        choices=["standard", "high_detail", "quick"],
        default="standard",
        help="Scanning mode to use"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        help="Output file path for results (JSON)"
    )
    
    parser.add_argument(
        "--version", 
        action="version",
        version=f"AI-Enhanced 3D Scanning System v{__version__}"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print(f"AI-Enhanced 3D Scanning Hearing Aid Manufacturing System v{__version__}")
    print("=" * 80)
    
    # Setup scan parameters
    scan_parameters = {
        "scan_mode": args.scan_mode,
        "timestamp": time.time()
    }
    
    # Run the pipeline
    try:
        results = run_pipeline(
            device_id=args.device,
            high_resolution=args.high_res,
            scan_parameters=scan_parameters
        )
        
        # Output results
        if args.output:
            import json
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        
        # Print summary
        if results["status"] == "success":
            print(f"Pipeline completed successfully in {results['pipeline_duration_seconds']:.2f} seconds")
            print(f"Captured {results['processing_results']['original_scan']['num_points']} points")
            print(f"Generated hearing aid model with volume: {results['processing_results']['model_results']['canal_measurements']['acoustic_pathway_volume']:.2f} cc")
        else:
            print(f"Pipeline failed: {results['error_message']}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.exception("Unhandled exception in main pipeline")
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
