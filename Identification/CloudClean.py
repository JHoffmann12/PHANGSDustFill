import logging
import os

import papermill as pm

logger = logging.getLogger(__name__)


def Remove(julia_path, julia_out_path, mask_path, orig_image_path, label_folder_path):
    """Execute the Julia source-removal notebook via papermill and return the output FITS path."""
    save_path = os.path.join(label_folder_path, "Source_Removal", "OriginalImageSourcesRemoved.fits")

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

    logger.info("Source removal complete → %s", save_path)
    return save_path