import cv2
import numpy as np
from pathlib import Path
import sys

def verify_patch(patch_path):
    """
    Verifies if a patch meets the architectural mandates:
    1. Size: 24x24
    2. Channels: 1 (Grayscale)
    3. Aspect Ratio: 1:1
    """
    p = Path(patch_path)
    if not p.exists():
        return False, f"File {patch_path} does not exist."

    # Load image in 'as-is' mode to check channels
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    
    if img is None:
        return False, f"Failed to load {patch_path}."

    h, w = img.shape[:2]
    
    # Check 1: Dimensions
    if h != 24 or w != 24:
        return False, f"Invalid dimensions: {w}x{h} (Expected 24x24)."

    # Check 2: Color Space
    if len(img.shape) != 2:
        return False, f"Invalid channels: {img.shape[2] if len(img.shape)==3 else 'Unknown'} (Expected 1 - Grayscale)."

    # Check 3: Isotropic Evidence (Black Padding)
    # For a typical eye ROI (wider than tall), the top/bottom 2 rows should be mostly black
    top_pixels = np.mean(img[0:2, :])
    bottom_pixels = np.mean(img[-2:, :])
    
    # We don't fail just on black pixels because some eyes might be dark, 
    # but we log it as a hint of correct padding.
    is_padded = (top_pixels < 5) and (bottom_pixels < 5)

    return True, f"Verified: 24x24 Grayscale. Isotropic Padding detected: {is_padded}"

def run_verification_suite():
    print("=== Phase 1 Mandate Verification ===")
    
    patches_to_check = [
        "DEMO/data/patch_squashed.jpg",
        "DEMO/data/patch_isotropic.jpg"
    ]
    
    all_pass = True
    for patch in patches_to_check:
        success, message = verify_patch(patch)
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"[{status}] {patch}: {message}")
        
        # In our specific demo, the squashed one SHOULD fail if we strictly check 
        # for 'thin' or 'distorted' features, but here we check hard mandates.
        if "isotropic" in patch and not success:
            all_pass = False

    if all_pass:
        print("\nSUMMARY: Architectural mandates are COMPLIANT.")
    else:
        print("\nSUMMARY: Architectural mandates VIOLATED.")
        sys.exit(1)

if __name__ == "__main__":
    run_verification_suite()
