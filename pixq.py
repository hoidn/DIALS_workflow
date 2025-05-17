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
    
    print(f"  Normalized s0_vec (from model): {s0_vec.tolist()}")
    
    # <<<< TEST: FORCE K_IN TO BE ALONG +Z AXIS >>>>
    s0_test_positive_z = np.array([0.0, 0.0, 1.0])
    print(f"  WARNING: TESTING WITH FORCED s0_vec = {s0_test_positive_z.tolist()}")
    
    k_magnitude = 2 * math.pi / wavelength
    k_in = s0_test_positive_z * k_magnitude
    # <<<< END TEST >>>>
    
    print(f"  Wavelength: {wavelength:.4f} Å")
    print(f"  Calculated k_in vector (FORCED +Z): ({k_in[0]:.4f}, {k_in[1]:.4f}, {k_in[2]:.4f}) Å⁻¹")
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

def verify_single_pixel_q(beam_model, panel_model, panel_idx_int, test_px_fast, test_py_slow):
    print(f"\n--- Verifying q for Panel Index {panel_idx_int} (Name: '{panel_model.get_name()}'), Pixel (fast={test_px_fast}, slow={test_py_slow}) ---")

    # 1. Get k_in
    k_in_vec_test, k_mag_scalar_test = calculate_incident_wavevector(beam_model) # Recalculate for clarity

    # 2. Get P_lab for this specific pixel using dxtbx direct method
    # panel.get_pixel_lab_coord() takes (fast_pixel_index, slow_pixel_index)
    P_lab_dxtbx = np.array(panel_model.get_pixel_lab_coord((test_px_fast, test_py_slow)))
    print(f"  P_lab (from dxtbx.get_pixel_lab_coord): {P_lab_dxtbx.tolist()} mm")

    # 3. Calculate k_out
    sample_origin_vec_test = np.array([0.0, 0.0, 0.0])
    D_scattered_dxtbx = P_lab_dxtbx - sample_origin_vec_test
    D_scattered_dxtbx_norm = np.linalg.norm(D_scattered_dxtbx)
    
    if D_scattered_dxtbx_norm < 1e-9:
        print("  Error: D_scattered_dxtbx_norm is zero.")
        return None

    s1_lab_dxtbx = D_scattered_dxtbx / D_scattered_dxtbx_norm
    k_out_dxtbx = s1_lab_dxtbx * k_mag_scalar_test
    print(f"  k_out (from dxtbx P_lab): {k_out_dxtbx.tolist()} Å⁻¹")

    # 4. Calculate q
    q_pixel_dxtbx_method = k_out_dxtbx - k_in_vec_test
    print(f"  q_pixel (dxtbx method): {q_pixel_dxtbx_method.tolist()} Å⁻¹")
    print(f"  |q_pixel| (dxtbx method): {np.linalg.norm(q_pixel_dxtbx_method):.4f} Å⁻¹")
    
    # 5. Compare with the value from your saved q-map (if available)
    try:
        # Use the integer panel index for loading, consistent with saving
        loaded_qx = np.load(f"panel_{panel_idx_int}_qmap_qx.npy")
        loaded_qy = np.load(f"panel_{panel_idx_int}_qmap_qy.npy")
        loaded_qz = np.load(f"panel_{panel_idx_int}_qmap_qz.npy")
        q_from_map = np.array([
            loaded_qx[test_py_slow, test_px_fast],
            loaded_qy[test_py_slow, test_px_fast],
            loaded_qz[test_py_slow, test_px_fast]
        ])
        print(f"  q_pixel (from saved map): {q_from_map.tolist()} Å⁻¹")
        print(f"  Difference (dxtbx_method - map): {(q_pixel_dxtbx_method - q_from_map).tolist()}")
        diff_norm = np.linalg.norm(q_pixel_dxtbx_method - q_from_map)
        print(f"  |Difference| (dxtbx_method - map): {diff_norm:.6e}")
    except FileNotFoundError:
        print(f"  Could not load saved q-maps for panel index {panel_idx_int} for comparison with this test pixel.")
    
    return q_pixel_dxtbx_method

if __name__ == "__main__":
    main() # Your existing main function call

    # --- Add the test ---
    print("\n--- Running Single Pixel Verification ---")
    # Ensure beam_model and detector_model are from the correct experiment file used in main()
    # If main() modified them, re-load or pass them carefully.
    # For simplicity, re-load here based on the global EXPERIMENT_FILE
    test_beam_model, test_detector_model = load_dials_models(EXPERIMENT_FILE)
    
    panel_to_test_idx = 0 # Test the first panel (index 0)
    if panel_to_test_idx < len(test_detector_model):
        panel_model_for_test = test_detector_model[panel_to_test_idx]
        # Test for Refl 0: Pixel (fast=2342, slow=30)
        q_verified = verify_single_pixel_q(test_beam_model, panel_model_for_test, panel_to_test_idx, 2342, 30)
        if q_verified is not None:
            # Compare this q_verified with q_bragg for Refl 0
            q_bragg_refl0 = np.array([0.4729, -0.6136, 0.4021]) # From consistency.py output
            print(f"\nFor Refl 0 (hkl: (-23, 17, 7)):")
            print(f"  q_bragg (manual)          : {q_bragg_refl0.tolist()}, |q|={np.linalg.norm(q_bragg_refl0):.4f}")
            print(f"  q_pixel (dxtbx verify)    : {q_verified.tolist()}, |q|={np.linalg.norm(q_verified):.4f}")
            print(f"  Diff (bragg - dxtbx verify): {(q_bragg_refl0 - q_verified).tolist()}")
            print(f"  |Diff|                    : {np.linalg.norm(q_bragg_refl0 - q_verified):.6f}")
    else:
        print(f"Could not find panel with index {panel_to_test_idx} to test.")
