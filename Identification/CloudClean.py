import papermill as pm
import os 

def Remove(julia_path, julia_out_path, mask_path, orig_image_path, label_folder_path): 
    save_path = os.path.join(label_folder_path, "Source_Removal")
    save_path = os.path.join(save_path, "OriginalImageSourcesRemoved.fits")

    pm.execute_notebook(
        julia_path,         # input_notebook_path
        julia_out_path,     # output_notebook_path
        parameters={
            "mask_path": mask_path,
            "image_path": orig_image_path,
            "save_path": save_path
        },
        kernel_name="julia-1.11"
    )

    return save_path