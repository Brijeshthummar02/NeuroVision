import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import app

def test_process_dicom_file():
    # Set up mock dataset
    mock_ds = MagicMock()
    mock_ds.PatientAge = "045Y"
    mock_ds.PatientSex = "M"
    mock_ds.Modality = "MR"
    mock_ds.MagneticFieldStrength = 3.0
    mock_ds.ScanningSequence = "SE"
    mock_ds.Manufacturer = "SIEMENS"
    mock_ds.StudyDate = "20260712"
    
    # 2D grayscale array of size 64x64
    mock_ds.pixel_array = np.full((64, 64), 128, dtype=np.uint16)
    
    with patch("pydicom.dcmread", return_value=mock_ds):
        img_bgr, metadata = app.process_dicom_file("mock_file.dcm")
        
        # Verify BGR conversion
        assert img_bgr.shape == (64, 64, 3)
        assert img_bgr.dtype == np.uint8
        
        # Verify metadata extraction & normalization
        assert metadata["patient_age"] == "45"
        assert metadata["patient_sex"] == "M"
        assert metadata["modality"] == "MR"
        assert metadata["magnetic_field_strength"] == "3.0T"
        assert metadata["scanning_sequence"] == "SE"
        assert metadata["manufacturer"] == "SIEMENS"
        assert metadata["study_date"] == "20260712"

def test_process_dicom_file_missing_fields():
    # Set up mock dataset with no metadata attributes
    # Delete optional properties to test missing field fallbacks
    mock_ds = MagicMock()
    del mock_ds.PatientAge
    del mock_ds.PatientSex
    del mock_ds.Modality
    del mock_ds.MagneticFieldStrength
    del mock_ds.ScanningSequence
    del mock_ds.Manufacturer
    del mock_ds.StudyDate
    
    mock_ds.pixel_array = np.full((64, 64), 100, dtype=np.uint16)
    
    with patch("pydicom.dcmread", return_value=mock_ds):
        img_bgr, metadata = app.process_dicom_file("mock_file.dcm")
        
        # Verify it falls back to 'N/A' safely without throwing AttributeErrors
        assert metadata["patient_age"] == "N/A"
        assert metadata["patient_sex"] == "N/A"
        assert metadata["modality"] == "N/A"
        assert metadata["magnetic_field_strength"] == "N/A"
        assert metadata["scanning_sequence"] == "N/A"
        assert metadata["manufacturer"] == "N/A"
        assert metadata["study_date"] == "N/A"
