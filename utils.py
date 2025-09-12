"""
Common utilities for ICASSP project
"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import re
from glob import glob, escape as glob_escape
from skimage.metrics import structural_similarity as ssim
import math

class ImageProcessor:
    """Image processing utilities"""
    
    @staticmethod
    def normalize_scale(img: np.ndarray, power_factor: float = 0.75) -> np.ndarray:
        """Normalize and scale image for better contrast"""
        img = np.array(img)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        if len(img.shape) == 2:
            img = np.power(img, 0.5)
        else:
            img = np.power(img, power_factor)
        
        return np.clip(img * 255, 0, 255).astype(np.uint8)
    
    @staticmethod
    def to_unit_range(arr: np.ndarray, is_gt_hint: bool = False) -> np.ndarray:
        """Convert array to [0,1] range"""
        a = arr.astype(np.float32)
        if is_gt_hint or a.max() > 1.5:  # likely 0..255 image
            a = a / 255.0
        return np.clip(a, 0.0, 1.0)
    
    @staticmethod
    def load_image(path: Union[str, Path], grayscale: bool = True) -> np.ndarray:
        """Load image from file"""
        path = Path(path)
        if path.suffix.lower() == '.npy':
            return np.load(path).astype(np.float32)
        
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        
        if grayscale and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return img.astype(np.float32)

class MetricsCalculator:
    """Calculate various metrics for image comparison"""
    
    @staticmethod
    def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Root Mean Squared Error"""
        return torch.sqrt(F.mse_loss(pred, target)).item()
    
    @staticmethod
    def compute_metrics(pred: np.ndarray, gt: np.ndarray, 
                       resample_to: str = "gt") -> Tuple[float, float, float, float]:
        """Compute SSIM, RMSE, PSNR, MAE metrics"""
        # Align shapes if needed
        if pred.shape != gt.shape:
            if resample_to == "gt":
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), 
                                interpolation=cv2.INTER_LINEAR)
            else:
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), 
                              interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0,1]
        pred_n = ImageProcessor.to_unit_range(pred, is_gt_hint=False)
        gt_n = ImageProcessor.to_unit_range(gt, is_gt_hint=True)
        
        # Calculate metrics
        ssim_val = float(ssim(pred_n, gt_n, data_range=1.0))
        mse = float(np.mean((pred_n - gt_n)**2))
        rmse = math.sqrt(mse)
        psnr = 10.0 * math.log10(1.0 / (mse + 1e-12))
        mae = float(np.mean(np.abs(pred_n - gt_n)))
        
        return ssim_val, rmse, psnr, mae

class FileManager:
    """File management utilities"""
    
    # Pattern for parsing filenames
    FILENAME_PATTERN = re.compile(
        r"^(?P<subject>\d+)_(?:ir|lr|pm)_(?P<condition>[A-Za-z0-9]+)_"
        r"image_(?P<frame>\d+)\.(?:png|jpg|jpeg|tif|tiff|bmp|gif|webp|npy)$",
        re.IGNORECASE
    )
    
    @staticmethod
    def list_files(directory: Union[str, Path], extension: str) -> List[str]:
        """List all files with given extension in directory"""
        base = glob_escape(str(Path(directory)))
        return sorted(glob(f"{base}/**/*.{extension}", recursive=True))
    
    @staticmethod
    def parse_filename(filename: str) -> Optional[Tuple[int, str, int]]:
        """Parse filename to extract subject, condition, frame"""
        match = FileManager.FILENAME_PATTERN.match(Path(filename).name)
        if not match:
            return None
        return (
            int(match.group('subject')),
            match.group('condition').lower(),
            int(match.group('frame'))
        )
    
    @staticmethod
    def build_file_map(directory: Union[str, Path], extension: str) -> Dict[Tuple, str]:
        """Build mapping from (subject, condition, frame) to filepath"""
        mapping = {}
        for filepath in FileManager.list_files(directory, extension):
            key = FileManager.parse_filename(filepath)
            if key:
                mapping[key] = filepath
        return mapping
    
    @staticmethod
    def find_paired_files(dir1: Union[str, Path], ext1: str,
                         dir2: Union[str, Path], ext2: str) -> List[Tuple[str, str]]:
        """Find paired files between two directories based on filename keys"""
        map1 = FileManager.build_file_map(dir1, ext1)
        map2 = FileManager.build_file_map(dir2, ext2)
        
        common_keys = sorted(set(map1.keys()) & set(map2.keys()))
        return [(map1[k], map2[k]) for k in common_keys]

class ManifestManager:
    """Manage manifest files for experiment tracking"""
    
    @staticmethod
    def create_manifest(run_name: str, run_dir: Union[str, Path], 
                       pred_dir: Union[str, Path], **kwargs) -> Dict:
        """Create manifest dictionary"""
        manifest = {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "pred_dir": str(pred_dir),
            "pred_ext": kwargs.get("pred_ext", "npy"),
            "gt_dir": kwargs.get("gt_dir", ""),
            "gt_ext": kwargs.get("gt_ext", "png"),
            "ir_dir": kwargs.get("ir_dir", ""),
            "ir_ext": kwargs.get("ir_ext", "png"),
            "resample_to": kwargs.get("resample_to", "gt")
        }
        return manifest
    
    @staticmethod
    def save_manifest(manifest: Dict, path: Union[str, Path]):
        """Save manifest to JSON file"""
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    @staticmethod
    def load_manifest(path: Union[str, Path]) -> Dict:
        """Load manifest from JSON file"""
        import json
        with open(path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def find_latest_manifest(base_dir: Union[str, Path]) -> Optional[str]:
        """Find the most recent manifest.json file"""
        root = Path(base_dir)
        if not root.exists():
            return None
        
        candidates = sorted(
            root.rglob("manifest.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return str(candidates[0]) if candidates else None