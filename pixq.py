import numpy as np
import math
import time
from dxtbx.model.experiment_list import ExperimentListFactory

# --- Configuration ---
EXPERIMENT_FILE = "indexed.expt" # Or indexed.expt
OUTPUT_FORMAT = "npy" # "npy" or "hdf5"
# ---------------------

def load_dials_models(experiment_file_path):
    """Loads beam and detector models from a DIALS experiment file."""
    print(f"Loading experiments from: {experiment_file_path}")
    experiments = ExperimentListFactory.from_json_file(experiment_file_path)
    if not experiments:
        raise ValueError(f"No experiments found in {experiment_file_path}")
    # Assuming one experiment for simplicity here, adapt if multiple
    experiment = experiments[0]
    return experiment.beam, experiment.detector

def calculate_incident_wavevector(beam_model):
    """Calculates the incident wavevector k_in."""
    wavelength = beam_model.get_wavelength()
    s0_raw = np.array(beam_model.get_s0())
    print(f"  DEBUG: Raw beam.get_s0() from dxtbx: {s0_raw.tolist()}")
    
    s0_norm = np.linalg.norm(s0_raw)
    if np.abs(s0_norm - 1.0) > 1e-6:  # If not close to unit vector
        print(f"  WARNING: Raw s0 magnitude is {s0_norm:.6f}, normalizing.")
        if s0_norm == 0:
            raise ValueError("s0 vector from beam model has zero length!")
        s0_vec = s0_raw / s0_norm
    else:
        s0_vec = s0_raw
        
    k_magnitude = 2 * math.pi / wavelength
    k_in = s0_vec * k_magnitude
    
    print(f"  Normalized s0_vec: {s0_vec.tolist()}")
    print(f"  Wavelength: {wavelength:.4f} Å")
    print(f"  Calculated k_in vector: ({k_in[0]:.4f}, {k_in[1]:.4f}, {k_in[2]:.4f}) Å⁻¹")
    return k_in, k_magnitude

def calculate_q_for_panel(panel_model, k_in_vec, k_mag_scalar, sample_origin_vec):
    """Calculates qx, qy, qz, and |q| maps for a single detector panel."""
    panel_origin_dxtbx = panel_model.get_origin()
    fast_axis_dxtbx = panel_model.get_fast_axis()
    slow_axis_dxtbx = panel_model.get_slow_axis()
    pixel_size_fast, pixel_size_slow = panel_model.get_pixel_size()
    num_fast, num_slow = panel_model.get_image_size()

    print(f"    Panel Dimensions: {num_fast} (fast) x {num_slow} (slow) pixels")
    
    # --- For Debugging Panel Geometry ---
    print(f"    DEBUG Panel (Name: '{panel_model.get_name()}'): Origin (mm): {panel_origin_dxtbx}")
    print(f"    DEBUG Panel (Name: '{panel_model.get_name()}'): Fast Axis: {fast_axis_dxtbx}")
    print(f"    DEBUG Panel (Name: '{panel_model.get_name()}'): Slow Axis: {slow_axis_dxtbx}")
    # -------------------------------------
    
    panel_origin = np.array(panel_origin_dxtbx)
    fast_axis = np.array(fast_axis_dxtbx)
    slow_axis = np.array(slow_axis_dxtbx)

    px_indices = np.arange(num_fast)
    py_indices = np.arange(num_slow)
    dist_fast_mm = (px_indices + 0.5) * pixel_size_fast
    dist_slow_mm = (py_indices + 0.5) * pixel_size_slow
    dist_fast_mm_grid, dist_slow_mm_grid = np.meshgrid(dist_fast_mm, dist_slow_mm)

    term_fast = dist_fast_mm_grid[..., np.newaxis] * fast_axis[np.newaxis, np.newaxis, :]
    term_slow = dist_slow_mm_grid[..., np.newaxis] * slow_axis[np.newaxis, np.newaxis, :]
    P_lab_all_pixels = panel_origin[np.newaxis, np.newaxis, :] + term_fast + term_slow

    D_scattered_all_pixels = P_lab_all_pixels - sample_origin_vec[np.newaxis, np.newaxis, :]
    D_scattered_norms = np.linalg.norm(D_scattered_all_pixels, axis=2)
    epsilon = 1e-9
    D_scattered_norms_safe = np.maximum(D_scattered_norms, epsilon)
    s1_lab_all_pixels = D_scattered_all_pixels / D_scattered_norms_safe[..., np.newaxis]
    k_out_all_pixels = s1_lab_all_pixels * k_mag_scalar
    
    q_all_pixels = k_out_all_pixels - k_in_vec[np.newaxis, np.newaxis, :]

    qx_map = q_all_pixels[:, :, 0]
    qy_map = q_all_pixels[:, :, 1]
    qz_map = q_all_pixels[:, :, 2]
    q_mag_map = np.linalg.norm(q_all_pixels, axis=2)
    
    return qx_map, qy_map, qz_map, q_mag_map

def save_q_maps(panel_id_str, qx_map, qy_map, qz_map, q_mag_map, output_format_str, metadata_dict=None):
    """Saves the calculated q-maps to file."""
    base_filename = f"panel_{panel_id_str}_qmap"
    if output_format_str == "npy":
        np.save(f"{base_filename}_qx.npy", qx_map)
        np.save(f"{base_filename}_qy.npy", qy_map)
        np.save(f"{base_filename}_qz.npy", qz_map)
        np.save(f"{base_filename}_mag.npy", q_mag_map)
        print(f"    Saved q-maps for panel {panel_id_str} to .npy files (prefix: {base_filename})")
    elif output_format_str == "hdf5":
        # import h5py # Ensure imported if using
        # with h5py.File(f"{base_filename}.h5", "w") as hf:
        #     hf.create_dataset("qx", data=qx_map, compression="gzip")
        #     hf.create_dataset("qy", data=qy_map, compression="gzip")
        #     hf.create_dataset("qz", data=qz_map, compression="gzip")
        #     hf.create_dataset("q_magnitude", data=q_mag_map, compression="gzip")
        #     if metadata_dict:
        #         for key, value in metadata_dict.items():
        #             try:
        #                 hf.attrs[key] = value
        #             except TypeError: # e.g. if value is a complex object not suitable for attr
        #                 print(f"      Warning: Could not save metadata '{key}' to HDF5 attributes.")
        # print(f"    Saved q-maps for panel {panel_id_str} to HDF5 file: {base_filename}.h5")
        print(f"    HDF5 output is commented out. To enable, uncomment the HDF5 block and `import h5py`.")

    else:
        print(f"    Unknown OUTPUT_FORMAT: {output_format_str}. Not saving files.")

def main():
    beam_model, detector_model = load_dials_models(EXPERIMENT_FILE)
    
    print("\nCalculating incident wavevector k_in:")
    k_in_vec, k_mag_scalar = calculate_incident_wavevector(beam_model)
    
    sample_origin_vec = np.array([0.0, 0.0, 0.0]) # Sample at lab origin

    total_pixels_processed = 0
    start_time_total = time.time()

    for panel_id, panel_model in enumerate(detector_model):
        panel_start_time = time.time()
        # panel.get_name() often returns a string like "0", "1", etc.
        # or could be an empty string if not explicitly named.
        # Using panel_id (int) for filenames is safer.
        panel_id_str = str(panel_id) 
        print(f"\nProcessing Panel ID {panel_id_str} (Name: '{panel_model.get_name()}'):")

        qx_map, qy_map, qz_map, q_mag_map = calculate_q_for_panel(
            panel_model, k_in_vec, k_mag_scalar, sample_origin_vec
        )
        
        # Prepare metadata for HDF5 if used
        metadata = {
            'wavelength_A': beam_model.get_wavelength(),
            'k_in_A_inv': k_in_vec.tolist(),
            'panel_origin_mm': list(panel_model.get_origin()),
            'pixel_size_mm': list(panel_model.get_pixel_size()),
            'image_size_pixels': list(panel_model.get_image_size())
        }
        save_q_maps(panel_id_str, qx_map, qy_map, qz_map, q_mag_map, OUTPUT_FORMAT, metadata_dict=metadata)

        panel_time = time.time() - panel_start_time
        num_fast, num_slow = panel_model.get_image_size()
        total_pixels_processed += num_fast * num_slow
        print(f"    Panel {panel_id_str} processing time: {panel_time:.2f} seconds.")

    total_time = time.time() - start_time_total
    print(f"\nFinished processing all panels.")
    print(f"Total pixels processed: {total_pixels_processed}")
    print(f"Total execution time: {total_time:.2f} seconds.")
    if total_pixels_processed > 0 :
        print(f"Time per megapixel: {total_time / (total_pixels_processed / 1e6):.2f} seconds/Mpixel.")

if __name__ == "__main__":
    main()
