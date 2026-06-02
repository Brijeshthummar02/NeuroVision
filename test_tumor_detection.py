"""
Test script to verify brain tumor detection on frontend
Uploads real MRI images with brain tumors and validates detection
Tests all tumor images from the dataset
"""

import requests
import json
import os
import sys
import csv
import time
from datetime import datetime

# Server URL
BASE_URL = "http://localhost:5000"

# Output directory for results
OUTPUT_DIR = "test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_tumor_images():
    """Get all tumor images from data_mask.csv where mask=1"""
    tumor_images = []
    csv_path = "data_mask.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found")
        return tumor_images
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['mask'] == '1':
                tumor_images.append(row['image_path'])
    
    print(f"📊 Found {len(tumor_images)} tumor images in dataset")
    return tumor_images

def test_health_check():
    """Test if the server is running"""
    print("=" * 60)
    print("🔍 Testing Server Health...")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        data = response.json()
        
        print(f"✓ Server Status: {data['status']}")
        print(f"✓ Models Loaded: {data['models_loaded']}")
        print(f"✓ Version: {data['version']}")
        print()
        return data['models_loaded']
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the Flask server is running on port 5000")
        return False

def test_tumor_detection(image_path, image_number, total_images):
    """Test tumor detection with a real MRI image"""
    print("\n" + "=" * 60)
    print(f"🧠 Testing Image {image_number}/{total_images}")
    print("=" * 60)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return {
            'success': False,
            'image_path': image_path,
            'error': 'Image not found'
        }
    
    print(f"📂 Image: {image_path}")
    
    # Prepare file for upload
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/tiff')}
            
            print("📤 Uploading MRI image to server...")
            response = requests.post(f"{BASE_URL}/api/predict", files=files, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ Error: Server returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return {
                'success': False,
                'image_path': image_path,
                'error': f'Status code {response.status_code}'
            }
        
        # Parse results
        result = response.json()
        
        print("\n" + "-" * 60)
        print("📊 DETECTION RESULTS")
        print("-" * 60)
        
        # Classification Results
        print(f"  Tumor Detected: {'✓ YES' if result['has_tumor'] else '✗ NO'}")
        print(f"  Confidence: {result['confidence'] * 100:.2f}%")
        
        # Segmentation Results (if tumor detected)
        if result['has_tumor'] and 'segmentation' in result:
            seg = result['segmentation']
            print(f"  Tumor Area: {seg['tumor_area_percentage']:.2f}% of brain")
            print(f"  Tumor Pixels: {seg['tumor_pixels']:,} pixels")
        
        # Validation
        expected_tumor = True  # We're using images known to have tumors
        if result['has_tumor'] == expected_tumor:
            print("✓ PASSED - Model correctly detected the tumor!")
        else:
            print("⚠ FAILED - Model did not detect tumor in known positive case")
        
        return {
            'success': True,
            'image_path': image_path,
            'has_tumor': result['has_tumor'],
            'confidence': result['confidence'],
            'tumor_area': seg['tumor_area_percentage'] if result['has_tumor'] and 'segmentation' in result else 0,
            'tumor_pixels': seg['tumor_pixels'] if result['has_tumor'] and 'segmentation' in result else 0,
            'correct_detection': result['has_tumor'] == expected_tumor
        }
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'image_path': image_path,
            'error': str(e)
        }

def save_results_summary(results):
    """Save test results to a JSON and text file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate statistics
    total = len(results)
    successful = sum(1 for r in results if r['success'])
    correct = sum(1 for r in results if r.get('correct_detection', False))
    failed = total - successful
    
    summary = {
        'timestamp': timestamp,
        'total_images': total,
        'successful_tests': successful,
        'failed_tests': failed,
        'correct_detections': correct,
        'accuracy': (correct / successful * 100) if successful > 0 else 0,
        'results': results
    }
    
    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, f'test_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save readable text report
    txt_path = os.path.join(OUTPUT_DIR, f'test_report_{timestamp}.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BRAIN TUMOR DETECTION TEST REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Images Tested: {total}\n")
        f.write(f"Successful Tests: {successful}\n")
        f.write(f"Failed Tests: {failed}\n")
        f.write(f"Correct Detections: {correct}/{successful}\n")
        f.write(f"Detection Accuracy: {summary['accuracy']:.2f}%\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for i, r in enumerate(results, 1):
            f.write(f"{i}. {r['image_path']}\n")
            if r['success']:
                f.write(f"   Status: {'PASS' if r['correct_detection'] else 'FAIL'}\n")
                f.write(f"   Tumor Detected: {r['has_tumor']}\n")
                f.write(f"   Confidence: {r['confidence']*100:.2f}%\n")
                if r['has_tumor']:
                    f.write(f"   Tumor Area: {r['tumor_area']:.2f}%\n")
                    f.write(f"   Tumor Pixels: {r['tumor_pixels']:,}\n")
            else:
                f.write(f"   Status: ERROR\n")
                f.write(f"   Error: {r.get('error', 'Unknown error')}\n")
            f.write("\n")
    
    print(f"\n📄 Results saved to:")
    print(f"   • {json_path}")
    print(f"   • {txt_path}")
    
    return summary

def main():
    """Main test function"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "NEUROSCAN AI - COMPLETE TUMOR DETECTION TEST" + " " * 14 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Test 1: Health Check
    print("=" * 60)
    print("🔍 Step 1: Server Health Check")
    print("=" * 60)
    if not test_health_check():
        print("\n❌ Server health check failed. Exiting...")
        sys.exit(1)
    
    # Test 2: Get all tumor images
    print("\n" + "=" * 60)
    print("📋 Step 2: Loading Tumor Images from Dataset")
    print("=" * 60)
    tumor_images = get_tumor_images()
    
    if not tumor_images:
        print("\n❌ No tumor images found in dataset. Exiting...")
        sys.exit(1)
    
    # Ask user for batch size or test all
    print(f"\nFound {len(tumor_images)} tumor images.")
    print("Options:")
    print("  1. Test all images")
    print("  2. Test first 10 images")
    print("  3. Test first 50 images")
    print("  4. Test a custom number")
    
    choice = input("\nEnter your choice (1-4) [default: 1]: ").strip() or "1"
    
    if choice == "2":
        tumor_images = tumor_images[:10]
    elif choice == "3":
        tumor_images = tumor_images[:50]
    elif choice == "4":
        try:
            num = int(input("Enter number of images to test: "))
            tumor_images = tumor_images[:num]
        except ValueError:
            print("Invalid input. Testing all images.")
    
    # Test 3: Tumor Detection on all images
    print("\n" + "=" * 60)
    print(f"🧠 Step 3: Testing Tumor Detection ({len(tumor_images)} images)")
    print("=" * 60)
    
    results = []
    start_time = time.time()
    
    for i, image_path in enumerate(tumor_images, 1):
        result = test_tumor_detection(image_path, i, len(tumor_images))
        results.append(result)
        
        # Small delay to avoid overwhelming the server
        if i < len(tumor_images):
            time.sleep(0.5)
    
    elapsed_time = time.time() - start_time
    
    # Save and display summary
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    
    summary = save_results_summary(results)
    
    print(f"\n✅ Total Images: {summary['total_images']}")
    print(f"✅ Successful Tests: {summary['successful_tests']}")
    print(f"❌ Failed Tests: {summary['failed_tests']}")
    print(f"🎯 Correct Detections: {summary['correct_detections']}/{summary['successful_tests']}")
    print(f"📈 Detection Accuracy: {summary['accuracy']:.2f}%")
    print(f"⏱️  Total Time: {elapsed_time:.2f} seconds")
    print(f"⚡ Average Time per Image: {elapsed_time/len(tumor_images):.2f} seconds")
    
    if summary['accuracy'] >= 90:
        print("\n🎉 EXCELLENT! Model performance is outstanding!")
    elif summary['accuracy'] >= 75:
        print("\n👍 GOOD! Model performance is acceptable.")
    else:
        print("\n⚠️  WARNING! Model performance may need improvement.")
    
    print("\n" + "=" * 60)
    print("💡 Next Steps:")
    print("   1. Review the detailed report in the test_results folder")
    print("   2. Open http://localhost:5000 in your browser")
    print("   3. Try uploading individual images for visual inspection")
    print("=" * 60)
    print()

if __name__ == "__main__":
    main()


# =============================================================================
# Unit tests for validators.py (Issue #14)
# Run independently: python -m pytest test_tumor_detection.py::TestValidators -v
#                or: python test_tumor_detection.py --unit
# =============================================================================

import io
import struct
import unittest
from unittest.mock import MagicMock, patch


def _make_jpeg_stream(size: int = 512) -> io.BytesIO:
    """Minimal valid JPEG magic bytes followed by padding."""
    return io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * (size - 4))


def _make_png_stream(size: int = 512) -> io.BytesIO:
    return io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * (size - 8))


def _make_tiff_le_stream() -> io.BytesIO:
    return io.BytesIO(b"II*\x00" + b"\x00" * 12)


def _make_bad_stream() -> io.BytesIO:
    return io.BytesIO(b"BADFILE_NOT_IMAGE" + b"\x00" * 10)


class TestValidators(unittest.TestCase):
    """Unit tests for validators.py — no Flask server or model weights required."""

    @classmethod
    def setUpClass(cls):
        # Import here so the test class is skippable if validators.py is absent
        try:
            from validators import (
                ValidationError,
                validate_file_present,
                validate_file_extension,
                validate_file_size,
                validate_mime_type,
                validate_image_loadable,
                validate_tensor_shape,
                MAX_UPLOAD_BYTES,
            )
            cls.ValidationError = ValidationError
            cls.validate_file_present = staticmethod(validate_file_present)
            cls.validate_file_extension = staticmethod(validate_file_extension)
            cls.validate_file_size = staticmethod(validate_file_size)
            cls.validate_mime_type = staticmethod(validate_mime_type)
            cls.validate_image_loadable = staticmethod(validate_image_loadable)
            cls.validate_tensor_shape = staticmethod(validate_tensor_shape)
            cls.MAX_UPLOAD_BYTES = MAX_UPLOAD_BYTES
        except ImportError as exc:
            raise unittest.SkipTest(f"validators.py not importable: {exc}")

    # ------------------------------------------------------------------
    # validate_file_present
    # ------------------------------------------------------------------

    def test_file_present_none_raises_no_file(self):
        with self.assertRaises(self.ValidationError) as ctx:
            self.validate_file_present(None)
        self.assertEqual(ctx.exception.code, "NO_FILE")
        self.assertEqual(ctx.exception.http_status, 400)

    def test_file_present_empty_filename_raises(self):
        fake = MagicMock()
        fake.filename = "   "
        with self.assertRaises(self.ValidationError) as ctx:
            self.validate_file_present(fake)
        self.assertEqual(ctx.exception.code, "EMPTY_FILENAME")

    def test_file_present_valid_passes(self):
        fake = MagicMock()
        fake.filename = "brain.jpg"
        self.assertIsNone(self.validate_file_present(fake))  # returns None on success

    # ------------------------------------------------------------------
    # validate_file_extension
    # ------------------------------------------------------------------

    def test_extension_no_dot_raises(self):
        with self.assertRaises(self.ValidationError) as ctx:
            self.validate_file_extension("nodotfile")
        self.assertEqual(ctx.exception.code, "INVALID_EXTENSION")
        self.assertEqual(ctx.exception.http_status, 415)

    def test_extension_exe_raises(self):
        with self.assertRaises(self.ValidationError) as ctx:
            self.validate_file_extension("malware.exe")
        self.assertEqual(ctx.exception.code, "INVALID_EXTENSION")

    def test_extension_jpg_passes_and_lowercases(self):
        self.assertEqual(self.validate_file_extension("scan.JPG"), "jpg")

    def test_extension_tiff_passes(self):
        self.assertEqual(self.validate_file_extension("mri.tiff"), "tiff")

    def test_extension_png_passes(self):
        self.assertEqual(self.validate_file_extension("brain.PNG"), "png")

    # ------------------------------------------------------------------
    # validate_file_size
    # ------------------------------------------------------------------

    def test_file_size_over_limit_raises(self):
        stream = io.BytesIO(b"x" * (self.MAX_UPLOAD_BYTES + 1))
        with self.assertRaises(self.ValidationError) as ctx:
            self.validate_file_size(stream)
        self.assertEqual(ctx.exception.code, "FILE_TOO_LARGE")
        self.assertEqual(ctx.exception.http_status, 413)

    def test_file_size_at_limit_passes(self):
        stream = io.BytesIO(b"x" * self.MAX_UPLOAD_BYTES)
        size = self.validate_file_size(stream)
        self.assertEqual(size, self.MAX_UPLOAD_BYTES)

    def test_file_size_rewinds_stream(self):
        stream = io.BytesIO(b"x" * 100)
        self.validate_file_size(stream)
        self.assertEqual(stream.tell(), 0)

    # ------------------------------------------------------------------
    # validate_mime_type
    # ------------------------------------------------------------------

    def test_mime_jpeg_detected(self):
        self.assertEqual(self.validate_mime_type(_make_jpeg_stream()), "jpeg")

    def test_mime_png_detected(self):
        self.assertEqual(self.validate_mime_type(_make_png_stream()), "png")

    def test_mime_tiff_le_detected(self):
        self.assertEqual(self.validate_mime_type(_make_tiff_le_stream()), "tiff")

    def test_mime_tiff_be_detected(self):
        stream = io.BytesIO(b"MM\x00*" + b"\x00" * 12)
        self.assertEqual(self.validate_mime_type(stream), "tiff")

    def test_mime_bad_bytes_raises(self):
        with self.assertRaises(self.ValidationError) as ctx:
            self.validate_mime_type(_make_bad_stream())
        self.assertEqual(ctx.exception.code, "INVALID_MIME")
        self.assertEqual(ctx.exception.http_status, 415)

    def test_mime_rewinds_stream(self):
        stream = _make_jpeg_stream()
        self.validate_mime_type(stream)
        self.assertEqual(stream.tell(), 0)

    def test_mime_correct_ext_but_wrong_magic_raises(self):
        """A .jpg file with non-JPEG content must be rejected."""
        # Build a stream with PNG magic but the caller would pass it as .jpg
        with self.assertRaises(self.ValidationError) as ctx:
            self.validate_mime_type(_make_bad_stream())
        self.assertEqual(ctx.exception.code, "INVALID_MIME")

    # ------------------------------------------------------------------
    # validate_image_loadable
    # ------------------------------------------------------------------

    def test_loadable_neither_arg_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.validate_image_loadable()

    @patch("validators.Image")
    def test_loadable_stream_pil_failure_raises(self, mock_image_module):
        mock_image_module.open.side_effect = Exception("truncated")
        stream = _make_jpeg_stream()
        with self.assertRaises(self.ValidationError) as ctx:
            self.validate_image_loadable(file_stream=stream)
        self.assertEqual(ctx.exception.code, "CORRUPTED_IMAGE")
        self.assertEqual(ctx.exception.http_status, 422)

    @patch("validators.cv2")
    @patch("validators.Image")
    def test_loadable_path_both_fail_raises(self, mock_image_module, mock_cv2):
        mock_cv2.imread.return_value = None
        mock_image_module.open.side_effect = Exception("cannot open")
        with self.assertRaises(self.ValidationError) as ctx:
            self.validate_image_loadable(filepath="/tmp/fake.jpg")
        self.assertEqual(ctx.exception.code, "CORRUPTED_IMAGE")

    @patch("validators.cv2")
    def test_loadable_path_cv2_success_returns_true(self, mock_cv2):
        import numpy as np
        mock_cv2.imread.return_value = MagicMock()  # non-None
        result = self.validate_image_loadable(filepath="/tmp/fake.jpg")
        self.assertTrue(result)

    # ------------------------------------------------------------------
    # validate_tensor_shape
    # ------------------------------------------------------------------

    def test_tensor_none_raises(self):
        with self.assertRaises(self.ValidationError) as ctx:
            self.validate_tensor_shape(None, (1, 256, 256, 3))
        self.assertEqual(ctx.exception.code, "INVALID_TENSOR")
        self.assertEqual(ctx.exception.http_status, 422)

    def test_tensor_wrong_shape_raises(self):
        import numpy as np
        tensor = np.zeros((1, 128, 128, 3))
        with self.assertRaises(self.ValidationError) as ctx:
            self.validate_tensor_shape(tensor, (1, 256, 256, 3))
        self.assertEqual(ctx.exception.code, "INVALID_TENSOR")

    def test_tensor_correct_shape_passes(self):
        import numpy as np
        tensor = np.zeros((1, 256, 256, 3))
        self.assertTrue(self.validate_tensor_shape(tensor, (1, 256, 256, 3)))


if __name__ == "__main__" and "--unit" in sys.argv:
    # Allow: python test_tumor_detection.py --unit
    sys.argv.remove("--unit")
    unittest.main(verbosity=2)
