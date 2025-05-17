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
    """
    Return s0 and its magnitude **without** the 2π factor,
    matching DIALS' internal convention.
    """
    wavelength = beam_model.get_wavelength()
    s0 = np.array(beam_model.get_s0())  # Already has length 1/λ
    print(f"  DEBUG: Raw beam.get_s0() from dxtbx: {s0.tolist()}")
    
    k_magnitude = 1.0 / wavelength  # Should match |s0|
    print(f"  Wavelength: {wavelength:.4f} Å")
    print(f"  |s0| from model: {np.linalg.norm(s0):.6f}")
    print(f"  Expected k_magnitude (1/λ): {k_magnitude:.6f}")
    
    # Use s0 directly as k_in - it's already in the right units
    k_in = s0
    
    print(f"  Using k_in vector: ({k_in[0]:.4f}, {k_in[1]:.4f}, {k_in[2]:.4f}) Å⁻¹")
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
    k_out_dxtbx = s1_lab_dxtbx * k_mag_scalar_test  # Using 1/λ to match DIALS convention
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
            q_bragg_refl0 = np.array([0.4729, 0.6267, 0.3810]) # Updated to match correct q_bragg
            print(f"\nFor Refl 0 (hkl: (-23, 17, 7)):")
            print(f"  q_bragg (manual)          : {q_bragg_refl0.tolist()}, |q|={np.linalg.norm(q_bragg_refl0):.4f}")
            print(f"  q_pixel (dxtbx verify)    : {q_verified.tolist()}, |q|={np.linalg.norm(q_verified):.4f}")
            print(f"  Diff (bragg - dxtbx verify): {(q_bragg_refl0 - q_verified).tolist()}")
            print(f"  |Diff|                    : {np.linalg.norm(q_bragg_refl0 - q_verified):.6f}")
    else:
        print(f"Could not find panel with index {panel_to_test_idx} to test.")
        
    # --- Sanity Check: q-vector at direct beam center ---
    print("\n--- Sanity Check: q-vector at Direct Beam Center ---")
    # test_beam_model and test_detector_model are already loaded from EXPERIMENT_FILE
    
    if test_detector_model:
        panel_for_beam_test_idx = 0 # Assuming beam hits panel 0
        if panel_for_beam_test_idx < len(test_detector_model):
            panel_for_beam_test = test_detector_model[panel_for_beam_test_idx]
            
            try:
                # Get s0 from the beam model (already normalized if using your function)
                # We need s0 to ask the panel where this vector intersects it
                s0_for_beam_center_raw = np.array(test_beam_model.get_s0())
                s0_for_beam_center_norm = np.linalg.norm(s0_for_beam_center_raw)
                if s0_for_beam_center_norm == 0: raise ValueError("s0 for beam center is zero vector")
                s0_for_beam_center = s0_for_beam_center_raw / s0_for_beam_center_norm

                # panel.get_beam_centre(s0) returns (fast_mm, slow_mm) of intersection
                # in the panel's 2D coordinate system.
                beam_center_mm_fast, beam_center_mm_slow = panel_for_beam_test.get_beam_centre(s0_for_beam_center)
                print(f"  Beam center on Panel {panel_for_beam_test_idx} (dxtbx panel.get_beam_centre(s0)): ({beam_center_mm_fast:.3f}, {beam_center_mm_slow:.3f}) mm from panel origin")

                px_size_f, px_size_s = panel_for_beam_test.get_pixel_size()
                
                # Convert mm beam center (relative to panel origin, in panel frame) to pixel indices
                beam_px_f_idx = int(round((beam_center_mm_fast / px_size_f) - 0.5))
                beam_px_s_idx = int(round((beam_center_mm_slow / px_size_s) - 0.5))
                
                num_f_panel, num_s_panel = panel_for_beam_test.get_image_size()
                print(f"  Calculated beam center pixel (fast_idx, slow_idx): ({beam_px_f_idx}, {beam_px_s_idx})")

                if not (0 <= beam_px_f_idx < num_f_panel and 0 <= beam_px_s_idx < num_s_panel):
                    print("  Warning: Calculated beam center pixel is outside panel dimensions!")
                else:
                    # Load q-maps generated by pixq.py (using the original s0 from the model)
                    qbm_qx = np.load(f"panel_{panel_for_beam_test_idx}_qmap_qx.npy")
                    qbm_qy = np.load(f"panel_{panel_for_beam_test_idx}_qmap_qy.npy")
                    qbm_qz = np.load(f"panel_{panel_for_beam_test_idx}_qmap_qz.npy")
                    
                    q_at_beam_center = np.array([
                        qbm_qx[beam_px_s_idx, beam_px_f_idx], # q_maps are [slow_idx, fast_idx]
                        qbm_qy[beam_px_s_idx, beam_px_f_idx],
                        qbm_qz[beam_px_s_idx, beam_px_f_idx]
                    ])
                    print(f"  q_pixelmap at beam center: {q_at_beam_center.tolist()} Å⁻¹")
                    print(f"  |q_pixelmap| at beam center: {np.linalg.norm(q_at_beam_center):.6e} Å⁻¹")

            except Exception as e:
                print(f"  Error during beam center sanity check: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for the error
        else:
            print(f"Panel index {panel_for_beam_test_idx} out of range for detector.")
