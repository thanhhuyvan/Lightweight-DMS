import cv2
import numpy as np
import os
from pathlib import Path

def crop_isotropic(img, x, y, w, h, target_size=(24, 24)):
    """
    Crops an ROI and pads it to a 1:1 ratio before resizing to target_size.
    This prevents the 'squashed' effect.
    """
    # 1. Extract ROI
    roi = img[y:y+h, x:x+w]
    
    # 2. Determine padding
    max_dim = max(w, h)
    
    # 3. Create square canvas
    # Use 3 channels if input is color, else 1
    if len(img.shape) == 3:
        square = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    else:
        square = np.zeros((max_dim, max_dim), dtype=np.uint8)
        
    # 4. Center ROI on canvas
    dx = (max_dim - w) // 2
    dy = (max_dim - h) // 2
    square[dy:dy+h, dx:dx+w] = roi
    
    # 5. Resize to target
    resized = cv2.resize(square, target_size, interpolation=cv2.INTER_AREA)
    return resized, roi

def run_demo():
    # Setup paths
    demo_data = Path("DEMO/data")
    demo_data.mkdir(parents=True, exist_ok=True)
    
    # Pick a sample image from the project
    sample_path = Path("frame/failed_detections/0_partcipant2_frame_00015.jpg")
    if not sample_path.exists():
        print(f"Sample {sample_path} not found. Using a random noise image for demo.")
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    else:
        img = cv2.imread(str(sample_path))

    # Mock an "Eye ROI" (usually eyes are wide: 60x30)
    # This represents a typical 'squash' risk area
    ex, ey, ew, eh = 200, 200, 80, 30
    
    # Technique A: Simple Resize (Squashed)
    roi_raw = img[ey:ey+eh, ex:ex+ew]
    squashed = cv2.resize(roi_raw, (24, 24))
    
    # Technique B: Isotropic Padding (Preserved)
    isotropic, _ = crop_isotropic(img, ex, ey, ew, eh, (24, 24))
    
    # Grayscale for both (as per project mandate)
    squashed_gray = cv2.cvtColor(squashed, cv2.COLOR_BGR2GRAY)
    isotropic_gray = cv2.cvtColor(isotropic, cv2.COLOR_BGR2GRAY)
    
    # Save results
    cv2.imwrite(str(demo_data / "roi_original.jpg"), roi_raw)
    cv2.imwrite(str(demo_data / "patch_squashed.jpg"), squashed_gray)
    cv2.imwrite(str(demo_data / "patch_isotropic.jpg"), isotropic_gray)
    
    print("Demo Complete!")
    print(f"Original ROI size: {ew}x{eh} (Ratio {ew/eh:.2f}:1)")
    print("Saved to DEMO/data/:")
    print(" - roi_original.jpg")
    print(" - patch_squashed.jpg (Notice how the eye looks 'thin' or 'tall')")
    print(" - patch_isotropic.jpg (Notice how the eye shape is preserved with padding)")

if __name__ == "__main__":
    run_demo()
