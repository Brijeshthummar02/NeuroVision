"""
validators.py — MRI Upload Validation Module
GSSoC 2026 Issue #14: Add MRI Upload Validation & Better Error Handling

Provides ValidationError and a set of validator functions that guard the
upload pipeline before any preprocessing or model inference runs.

Validation order expected by app.py /api/predict:
    1. validate_file_present()
    2. validate_file_extension()
    3. validate_file_size()
    4. validate_mime_type()
    5. validate_image_loadable()          -- Pass 1: stream (before save)
    6. file.save(filepath)
    7. validate_image_loadable(filepath)  -- Pass 2: path  (after save)
    8. preprocess -> predict
    9. validate_tensor_shape()
"""

import os
import logging
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_MB", "16")) * 1024 * 1024

ALLOWED_EXTENSIONS: frozenset = frozenset({"png", "jpg", "jpeg", "tif", "tiff"})

# Maps a human-readable format name to its accepted magic-byte prefixes.
# Reading only the first 16 bytes is sufficient for all supported formats.
MAGIC_SIGNATURES: dict = {
    "jpeg": [b"\xff\xd8\xff"],
    "png":  [b"\x89PNG"],
    "tiff": [b"II*\x00", b"MM\x00*"],
}

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class ValidationError(Exception):
    """
    Raised when an uploaded file fails any validation check.

    Attributes:
        message     -- human-readable description (safe to surface to the user)
        code        -- machine-readable error code (e.g. "INVALID_MIME")
        http_status -- suggested HTTP status code for the API response
    """

    def __init__(self, message: str, code: str, http_status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.http_status = http_status


# ---------------------------------------------------------------------------
# Validator functions
# ---------------------------------------------------------------------------


def validate_file_present(file) -> None:
    """
    Ensure a file was actually included in the request.

    Args:
        file: werkzeug FileStorage object, or None if the key was absent.

    Raises:
        ValidationError: code="NO_FILE" (400) if file is None or has no name.
    """
    if file is None or not getattr(file, "filename", None):
        raise ValidationError(
            message="No file uploaded. Please attach an MRI image.",
            code="NO_FILE",
            http_status=400,
        )
    if file.filename.strip() == "":
        raise ValidationError(
            message="No file selected. Please choose an MRI image before uploading.",
            code="EMPTY_FILENAME",
            http_status=400,
        )


def validate_file_extension(filename: str) -> str:
    """
    Check that the file extension is in the accepted allowlist.

    Args:
        filename: original filename string from the upload.

    Returns:
        Lowercased file extension (without leading dot) on success.

    Raises:
        ValidationError: code="INVALID_EXTENSION" (415) for unsupported types.
    """
    if "." not in filename:
        raise ValidationError(
            message="File has no extension. Accepted formats: PNG, JPG, JPEG, TIF, TIFF.",
            code="INVALID_EXTENSION",
            http_status=415,
        )

    ext = filename.rsplit(".", 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            message=(
                f"Unsupported file type '.{ext}'. "
                "Accepted formats: PNG, JPG, JPEG, TIF, TIFF."
            ),
            code="INVALID_EXTENSION",
            http_status=415,
        )
    return ext


def validate_file_size(file_stream) -> int:
    """
    Ensure the upload does not exceed MAX_UPLOAD_BYTES.

    Seeks to the end of the stream to measure size, then rewinds to position 0
    so subsequent reads are unaffected.

    Args:
        file_stream: seekable file-like object (werkzeug FileStorage.stream).

    Returns:
        File size in bytes on success.

    Raises:
        ValidationError: code="FILE_TOO_LARGE" (413) if size exceeds limit.
    """
    file_stream.seek(0, 2)  # seek to end
    size = file_stream.tell()
    file_stream.seek(0)     # rewind

    max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    if size > MAX_UPLOAD_BYTES:
        size_mb = size / (1024 * 1024)
        raise ValidationError(
            message=(
                f"File size ({size_mb:.1f} MB) exceeds the maximum allowed "
                f"upload size of {max_mb} MB."
            ),
            code="FILE_TOO_LARGE",
            http_status=413,
        )
    return size


def validate_mime_type(file_stream) -> str:
    """
    Verify the file's actual content via magic-byte inspection.

    Reads the first 16 bytes of the stream and compares them against known
    signatures for JPEG, PNG, and TIFF. The stream is rewound to position 0
    after the check so it remains usable by the caller.

    Args:
        file_stream: seekable file-like object.

    Returns:
        Detected format string ("jpeg", "png", or "tiff") on success.

    Raises:
        ValidationError: code="INVALID_MIME" (415) if no signature matches.
    """
    file_stream.seek(0)
    header = file_stream.read(16)
    file_stream.seek(0)

    for fmt, signatures in MAGIC_SIGNATURES.items():
        for sig in signatures:
            if header.startswith(sig):
                logger.debug("validate_mime_type: detected format '%s'", fmt)
                return fmt

    raise ValidationError(
        message=(
            "File content does not match a recognised MRI image format. "
            "Please upload a valid PNG, JPEG, or TIFF file."
        ),
        code="INVALID_MIME",
        http_status=415,
    )


def validate_image_loadable(file_stream=None, filepath: str = None) -> bool:
    """
    Verify that the image can be decoded by both PIL and OpenCV.

    Two optional passes:
        Pass 1 (stream)  — call PIL.Image.open().verify() on an in-memory copy
                           of the stream before the file is written to disk.
        Pass 2 (filepath) — call cv2.imread() on the saved path after the file
                            has been persisted.

    At least one of file_stream or filepath must be provided.

    Args:
        file_stream: seekable file-like object (Pass 1). May be None to skip.
        filepath:    absolute path to the saved file (Pass 2). May be None to skip.

    Returns:
        True when all requested checks pass.

    Raises:
        ValidationError: code="CORRUPTED_IMAGE" (422) on any decode failure.
        ValueError: if neither argument is provided.
    """
    if file_stream is None and filepath is None:
        raise ValueError("validate_image_loadable requires file_stream or filepath.")

    # ------------------------------------------------------------------
    # Pass 1 — PIL stream verification (before disk write)
    # ------------------------------------------------------------------
    if file_stream is not None:
        try:
            file_stream.seek(0)
            buf = BytesIO(file_stream.read())
            file_stream.seek(0)

            img = Image.open(buf)
            img.verify()  # raises if the file is truncated or corrupt
        except Exception as exc:
            logger.warning("validate_image_loadable (stream): PIL verify failed — %s", exc)
            raise ValidationError(
                message=(
                    "The uploaded image appears to be corrupted or incomplete. "
                    "Please re-export the MRI scan and try again."
                ),
                code="CORRUPTED_IMAGE",
                http_status=422,
            ) from exc

    # ------------------------------------------------------------------
    # Pass 2 — OpenCV path verification (after disk write)
    # ------------------------------------------------------------------
    if filepath is not None:
        img_cv = cv2.imread(filepath)
        if img_cv is None:
            # Attempt PIL fallback before declaring failure
            try:
                with Image.open(filepath) as img_pil:
                    img_pil.load()
            except Exception as exc:
                logger.warning(
                    "validate_image_loadable (path): both cv2 and PIL failed for '%s' — %s",
                    filepath, exc,
                )
                raise ValidationError(
                    message=(
                        "The saved image file could not be opened. "
                        "It may be corrupted or in an unsupported colour mode."
                    ),
                    code="CORRUPTED_IMAGE",
                    http_status=422,
                ) from exc

    return True


def validate_tensor_shape(tensor: np.ndarray, expected: tuple) -> bool:
    """
    Assert that a preprocessed tensor has the expected shape before inference.

    Args:
        tensor:   numpy array produced by a preprocessing function.
        expected: tuple of expected dimensions, e.g. (1, 256, 256, 3).

    Returns:
        True if the shape matches.

    Raises:
        ValidationError: code="INVALID_TENSOR" (422) if shapes differ.
    """
    if tensor is None:
        raise ValidationError(
            message="Image preprocessing produced no output. The image may be invalid.",
            code="INVALID_TENSOR",
            http_status=422,
        )
    if tensor.shape != expected:
        raise ValidationError(
            message=(
                f"Unexpected image dimensions after preprocessing: "
                f"got {tensor.shape}, expected {expected}. "
                "Please ensure the image is a valid brain MRI scan."
            ),
            code="INVALID_TENSOR",
            http_status=422,
        )
    return True
