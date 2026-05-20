import logging
import os
import subprocess
import time

import papermill as pm

logger = logging.getLogger(__name__)


def _unlock_file(path):
    """On Windows, clear path so FITSIO can recreate it.

    Tries delete first; if the file is locked by a lingering Julia kernel,
    kills julia.exe and retries; if still locked, renames it out of the way
    (Windows allows renaming open files even when deletion fails).
    """
    if not os.path.exists(path):
        return
    try:
        os.remove(path)
        return
    except (PermissionError, OSError):
        pass

    logger.warning("Output file locked; killing lingering Julia processes: %s", path)
    for exe in ("julia.exe", "julia-1.11.exe"):
        subprocess.run(["taskkill", "/F", "/IM", exe], capture_output=True)
    time.sleep(3)

    try:
        os.remove(path)
        return
    except (PermissionError, OSError):
        pass

    # Windows allows renaming open files; move it aside so Julia can write fresh.
    stale = path + ".stale"
    for old in (stale, path + ".stale2"):
        if os.path.exists(old):
            try:
                os.remove(old)
            except Exception:
                pass
    try:
        os.rename(path, stale)
        logger.warning("Renamed locked file to %s; it will be removed on next successful run.", stale)
    except Exception as e:
        logger.error("Could not remove or rename locked file %s: %s", path, e)
        raise


def Remove(julia_path, julia_out_path, mask_path, orig_image_path, label_folder_path):
    """Execute the Julia source-removal notebook via papermill and return the output FITS path."""
    save_path = os.path.join(label_folder_path, "Source_Removal", "OriginalImageSourcesRemoved.fits")

    _unlock_file(save_path)

    logger.info("Running Julia source removal notebook: %s", julia_path)
    try:
        pm.execute_notebook(
            julia_path,
            julia_out_path,
            parameters={
                "mask_path": mask_path,
                "image_path": orig_image_path,
                "save_path": save_path,
                "widx": 275,
            },
            kernel_name="julia-1.11",
        )
    except Exception as e:
        logger.error("papermill execution failed: %s", e)
        raise

    logger.info("Source removal complete -> %s", save_path)
    return save_path