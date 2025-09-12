#!/usr/bin/env python
"""
Combined evaluation script for metrics calculation and baseline methods
"""
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import distance_transform_edt, uniform_filter
from sklearn.linear_model import Ridge
from skimage.filters import threshold_otsu
from skimage.morphology import closing, opening, disk
from tqdm import tqdm

from config import Config
from utils import ImageProcessor, MetricsCalculator, FileManager, ManifestManager

class EMAProcessor:
    """Exponential Moving Average processor for temporal smoothing"""
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.prev_ema = None
        self.prev_seq_id = None
    
    def process(self, frame: np.ndarray, seq_id: Optional[Tuple] = None) -> np.ndarray:
        """Apply EMA to frame"""
        if seq_id != self.prev_seq_id:
            self.prev_ema = None
            self.prev_seq_id = seq_id
        
        if self.prev_ema is None:
            ema = frame
        else:
            ema = self.alpha * frame + (1 - self.alpha) * self.prev_ema
        
        self.prev_ema = ema
        return ema

class BaselinePredictor:
    """Classical baseline methods for IR to PM prediction"""
    
    def __init__(self, config: Config):
        self.config = config
        self.ridge_model = None
        self.calibration_stats = {}
    
    def extract_features(self, ir: np.ndarray) -> np.ndarray:
        """Extract features from IR image"""
        # Gradient magnitude
        gx = cv2.Sobel(ir, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ir, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx*gx + gy*gy)
        
        # Local standard deviation
        k = 7
        mu = uniform_filter(ir, size=k)
        mu2 = uniform_filter(ir*ir, size=k)
        var = np.clip(mu2 - mu*mu, 0, None)
        local_std = np.sqrt(var)
        
        # Contact mask and distance transform
        mask = self._contact_mask(ir)
        dist = distance_transform_edt(mask)
        
        return np.stack([ir, grad_mag, local_std, dist], axis=-1)
    
    def _contact_mask(self, ir: np.ndarray) -> np.ndarray:
        """Create contact mask from IR image"""
        ir_n = (ir - ir.min()) / max(1e-6, (ir.max() - ir.min()))
        threshold = threshold_otsu(ir_n)
        mask = (ir_n >= threshold).astype(np.uint8)
        mask = closing(mask, disk(3))
        mask = opening(mask, disk(3))
        return mask
    
    def calibrate_ridge(self, ir_paths: List[str], pm_paths: List[str]) -> None:
        """Calibrate Ridge regression model"""
        sample_frac = self.config.get('baseline.sample_pix_frac')
        
        X_list, y_list = [], []
        for ir_path, pm_path in tqdm(zip(ir_paths, pm_paths), 
                                     desc="Calibrating Ridge", total=len(ir_paths)):
            ir = ImageProcessor.load_image(ir_path)
            pm = ImageProcessor.load_image(pm_path)
            
            if pm.shape != ir.shape:
                pm = cv2.resize(pm, (ir.shape[1], ir.shape[0]), 
                              interpolation=cv2.INTER_LINEAR)
            
            features = self.extract_features(ir).reshape(-1, 4)
            pm_flat = pm.reshape(-1)
            
            # Sample pixels
            n_pixels = features.shape[0]
            n_sample = max(1, int(n_pixels * sample_frac))
            indices = np