from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family import flex # For loading reflection table
import numpy as np
from scitbx import matrix # For matrix/vector operations with crystal model
import math

# --- Configuration ---
INDEXED_EXPT_FILE = "indexed_refined_detector.expt" # Changed
INDEXED_REFL_FILE = "indexed_refined_detector.refl" # Changed
# ---------------------

# --- Helper function to get q_bragg ---
def get_q_bragg_from_reflection(refl, experiment):
    """
    Return q_bragg in the DIALS laboratory frame (Å⁻¹) using the
    crystal model's hkl_to_reciprocal_space_vec method.
    This method handles all necessary frame transformations (C·S·F·A).
    DIALS units (1/λ).
    """
    hkl_vec = matrix.col(refl["miller_index"])      # Column vector (h, k, l)
    
    # experiment.crystal.hkl_to_reciprocal_space_vec returns a scitbx.matrix.col object
    q_bragg_lab_scitbx = experiment.crystal.hkl_to_reciprocal_space_vec(hkl_vec)
    
    q_bragg_np = np.array(q_bragg_lab_scitbx.elems) # Convert to NumPy array
    
    return q_bragg_np

# --- Helper function to calculate q_pixel for a specific pixel (like in pixq.py) ---
def calculate_q_for_single_pixel(beam_model, panel_model, px_fast_idx, py_slow_idx):
    """Calculates q-vector for a specific pixel directly using beam and detector models."""
    # 1. k_in - use s0 directly (already has length 1/λ)
    s0 = np.array(beam_model.get_s0())
    k_in = s0
    k_magnitude = np.linalg.norm(s0)  # Should be 1/λ

    # 2. P_lab for this specific pixel
    # panel.get_pixel_lab_coord() takes (fast_pixel_index, slow_pixel_index)
    P_lab = np.array(panel_model.get_pixel_lab_coord((px_fast_idx, py_slow_idx)))

    # 3. k_out
    sample_origin = np.array([0.0, 0.0, 0.0])
    D_scattered = P_lab - sample_origin
    D_scattered_norm = np.linalg.norm(D_scattered)
    if D_scattered_norm < 1e-9: return np.array([0.0, 0.0, 0.0]) # Should not happen for Bragg peak
    s1_lab = D_scattered / D_scattered_norm
    k_out = s1_lab * k_magnitude
    
    # 4. q
    q_pixel = k_out - k_in
    return q_pixel

# --- Main part ---
print(f"Loading indexed experiments from: {INDEXED_EXPT_FILE}")
experiments = ExperimentListFactory.from_json_file(INDEXED_EXPT_FILE)
print(f"Loading indexed reflections from: {INDEXED_REFL_FILE}")
reflections = flex.reflection_table.from_file(INDEXED_REFL_FILE)

indexed_sel = reflections.get_flags(reflections.flags.indexed)
reflections_indexed = reflections.select(indexed_sel)
print(f"Found {len(reflections_indexed)} indexed reflections.")

if len(reflections_indexed) == 0:
    print("No indexed reflections found. Cannot perform consistency check.")
    exit()

print("\nComparing q_bragg with directly recalculated q_pixel for indexed reflections:")
q_diff_magnitudes = []
q_bragg_mags_list = []
q_pixel_mags_list = []
all_px_coords = []
all_py_coords = []
# For comparing q_pixel_recalculated with DIALS' q_pred (s1-s0)
q_pixel_vs_dials_pred_diff_norms = flex.double()


for i in range(len(reflections_indexed)):
    refl = reflections_indexed[i]
    
    # Get experiment corresponding to this reflection's experiment ID
    # For single image processing, experiment_id is usually 0
    experiment_id = refl['id']
    if experiment_id < 0 or experiment_id >= len(experiments):
        print(f"Warning: Invalid experiment ID {experiment_id} for reflection {i}. Skipping.")
        continue
    current_experiment = experiments[experiment_id]

    # a. Calculate q_bragg
    q_bragg = get_q_bragg_from_reflection(refl, current_experiment) # (3,) NumPy array

    # b. Get pixel coordinates and panel ID for this Bragg peak
    panel_id = refl['panel']
    # xyzobs.px.value gives (x_px, y_px, frame_px)
    # For q-map lookup, we need integer pixel indices (fast_px_idx, slow_px_idx)
    # DIALS pixel coordinates are (fast_scan_coord, slow_scan_coord)
    # NumPy array indexing is typically (row_idx=slow_scan, col_idx=fast_scan)
    
    # b. Get MILLIMETER coordinates and panel ID for this Bragg peak from the refined values
    # 'xyzcal.mm' gives (x_mm, y_mm, frame_mm_equivalent_if_scan)
    # For stills, the third component might be less meaningful or 0.
    x_cal_mm, y_cal_mm, _ = refl['xyzcal.mm'] 

    # c. Convert mm coordinates on the panel to pixel indices
    # panel.millimeter_to_pixel() takes (fast_mm_coord, slow_mm_coord) *in the panel's 2D coordinate system*
    # We have x_cal_mm, y_cal_mm from refl['xyzcal.mm']. These are ALREADY
    # in the panel's 2D coordinate system (fast, slow directions).
    
    beam_model_current = current_experiment.beam # Ensure these are defined before try block
    detector_model_current = current_experiment.detector
    if panel_id >= len(detector_model_current):
        print(f"  Warning: Panel ID {panel_id} out of range for detector. Skipping reflection {i}.")
        continue
    panel_model_current = detector_model_current[panel_id]

    try:
        fast_px_cal, slow_px_cal = panel_model_current.millimeter_to_pixel((x_cal_mm, y_cal_mm))
    except RuntimeError as e:
        print(f"  Warning: Could not convert mm to pixel for reflection {i} (panel {panel_id}): {e}")
        print(f"    xyzcal.mm: ({x_cal_mm}, {y_cal_mm})")
        # Fallback or skip if necessary
        try:
            print("    Falling back to xyzobs.px.value for q_pixel calculation.")
            x_obs_px, y_obs_px, _ = refl['xyzobs.px.value']
            px_idx = int(round(x_obs_px))
            py_idx = int(round(y_obs_px))
        except KeyError:
            print(f"    xyzobs.px.value also not found. Skipping reflection {i}.")
            continue # Skip this reflection
    else: # If millimeter_to_pixel was successful
        px_idx = int(round(fast_px_cal))
        py_idx = int(round(slow_px_cal))
    
    # d. Recalculate q_pixel for this specific pixel using the current experiment's models
    q_pixel_recalculated = calculate_q_for_single_pixel(beam_model_current, panel_model_current, px_idx, py_idx)

    if i == 0: # Detailed print for the first reflection
        print(f"\n--- Detailed Debug for Refl {i} ---")
        hkl_tuple = refl['miller_index']
        print(f"  Miller Index (hkl): {hkl_tuple}")
        
        A_matrix_sqr = matrix.sqr(current_experiment.crystal.get_A())
        print(f"  A-matrix elements: {A_matrix_sqr.elems}")
        
        hkl_col = matrix.col(hkl_tuple)
        q_scitbx_manual = A_matrix_sqr * hkl_col
        q_bragg_np_unflipped = np.array(q_scitbx_manual.elems)
        print(f"  q_bragg (A*hkl, unflipped): {q_bragg_np_unflipped.tolist()}")
        
        q_bragg_np_flipped = q_bragg_np_unflipped.copy()
        q_bragg_np_flipped[1] *= -1
        print(f"  q_bragg (A*hkl, Y-flipped by script): {q_bragg_np_flipped.tolist()}") # This should match q_bragg variable
        
        print(f"  q_pixel_recalculated (for comparison): {q_pixel_recalculated.tolist()}")
        print(f"--- End Detailed Debug for Refl {i} ---\n")

    # d. Compare q_bragg (from A matrix) with q_pixel_recalculated
    q_difference = q_bragg - q_pixel_recalculated
    diff_magnitude = np.linalg.norm(q_difference)
    q_diff_magnitudes.append(diff_magnitude)
    all_px_coords.append(px_idx)
    all_py_coords.append(py_idx)

    q_bragg_mag = np.linalg.norm(q_bragg)
    q_pixel_mag = np.linalg.norm(q_pixel_recalculated)
    q_bragg_mags_list.append(q_bragg_mag)
    q_pixel_mags_list.append(q_pixel_mag)

    # e. Compare q_pixel_recalculated with DIALS' q_pred (s1-s0)
    s0_vec = np.array(current_experiment.beam.get_s0())
    s1_vec = np.array(refl['s1'])
    q_pred_dials = s1_vec - s0_vec
    
    diff_q_pixel_vs_dials_pred = q_pred_dials - q_pixel_recalculated
    q_pixel_vs_dials_pred_diff_norms.append(np.linalg.norm(diff_q_pixel_vs_dials_pred))

    if i < 10 or diff_magnitude > 0.01 : # Print first few and any large differences
        print(f"Refl {i} (hkl: {refl['miller_index']}):")
        print(f"  Panel: {panel_id}, Pixel (fast,slow): ({px_idx},{py_idx})")
        print(f"  q_bragg (from A matrix) : ({q_bragg[0]:.4f}, {q_bragg[1]:.4f}, {q_bragg[2]:.4f}), |q|={q_bragg_mag:.4f}")
        print(f"  q_pixel (recalculated)  : ({q_pixel_recalculated[0]:.4f}, {q_pixel_recalculated[1]:.4f}, {q_pixel_recalculated[2]:.4f}), |q|={q_pixel_mag:.4f}")
        print(f"  Difference vector       : ({q_difference[0]:.4f}, {q_difference[1]:.4f}, {q_difference[2]:.4f})")
        print(f"  |q_bragg - q_pixel_recalc|: {diff_magnitude:.6f} Å⁻¹")
        if q_bragg_mag > 1e-6 : # Avoid division by zero for central beam
             print(f"  Relative diff |q_bragg - q_pixel_recalc| / |q_bragg| : {diff_magnitude/q_bragg_mag:.6f}")


if q_diff_magnitudes:
    q_diff_magnitudes = np.array(q_diff_magnitudes)
    print("\nSummary of |q_bragg - q_pixel_recalculated| (Å⁻¹):")
    print(f"  Mean:   {np.mean(q_diff_magnitudes):.6f}")
    print(f"  Median: {np.median(q_diff_magnitudes):.6f}")
    print(f"  StdDev: {np.std(q_diff_magnitudes):.6f}")
    print(f"  Min:    {np.min(q_diff_magnitudes):.6f}")
    print(f"  Max:    {np.max(q_diff_magnitudes):.6f}")

    if len(q_pixel_vs_dials_pred_diff_norms) > 0:
        print(f"\nMean |q_pred_dials - q_pixel_recalculated|: {flex.mean(q_pixel_vs_dials_pred_diff_norms):.6f} Å⁻¹")

    # Optional: Plot a histogram of differences or a scatter plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(q_diff_magnitudes, bins=50)
        plt.xlabel("|q_bragg - q_pixel_recalculated| (Å⁻¹)")
        plt.ylabel("Frequency")
        plt.title("Distribution of q-vector Differences (Direct Recalculation)")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.scatter(q_bragg_mags_list, q_pixel_mags_list, alpha=0.5, s=10)
        min_val = min(min(q_bragg_mags_list), min(q_pixel_mags_list))
        max_val = max(max(q_bragg_mags_list), max(q_pixel_mags_list))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal Match") # y=x line
        plt.xlabel("|q_bragg| (Å⁻¹)")
        plt.ylabel("|q_pixel_recalculated| (Å⁻¹)")
        plt.title("Comparison of q-vector Magnitudes")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("q_consistency_check_direct_recalc.png")
        print("\nSaved q_consistency_check_direct_recalc.png")
        # plt.show() # Uncomment to display plot interactively

        # Create the heatmap
        if all_px_coords and all_py_coords: # Ensure lists are not empty
            all_px_coords_np = np.array(all_px_coords)
            all_py_coords_np = np.array(all_py_coords)
            all_q_diff_magnitudes_np = np.array(q_diff_magnitudes) # Already created

            try:
                panel0 = experiments[0].detector[0] # Assuming first experiment, first panel
                num_fast_plot, num_slow_plot = panel0.get_image_size()
            except IndexError: # Handle case where experiments or detector might be empty or not structured as expected
                print("Warning: Could not get panel dimensions for heatmap plot limits. Using max coordinates.")
                num_fast_plot = np.max(all_px_coords_np) + 1 if len(all_px_coords_np) > 0 else 2463 # Default fallback
                num_slow_plot = np.max(all_py_coords_np) + 1 if len(all_py_coords_np) > 0 else 2527 # Default fallback
            except Exception as e:
                print(f"Warning: Error getting panel dimensions for heatmap: {e}. Using max coordinates.")
                num_fast_plot = np.max(all_px_coords_np) + 1 if len(all_px_coords_np) > 0 else 2463
                num_slow_plot = np.max(all_py_coords_np) + 1 if len(all_py_coords_np) > 0 else 2527


            plt.figure(figsize=(10, 10))
            # Normalize error for color mapping, cap at a reasonable max_error for visualization
            # vmax_error = np.percentile(all_q_diff_magnitudes_np, 95) # Cap at 95th percentile
            vmax_error = 1.1 # Adjusted fixed cap based on recent typical max values, can be tuned
            if np.any(all_q_diff_magnitudes_np > vmax_error): # Check if any errors exceed the cap
                 print(f"Note: Some |Δq| values exceed heatmap vmax of {vmax_error} Å⁻¹ and will be shown as the max color.")


            scatter = plt.scatter(
                all_px_coords_np, 
                all_py_coords_np, 
                c=all_q_diff_magnitudes_np, 
                cmap='viridis', # Or 'hot', 'coolwarm', etc.
                s=10, # Adjust marker size
                vmin=0, 
                vmax=vmax_error # Cap the color scale for better visibility of smaller errors
            )
            
            plt.colorbar(scatter, label='|Δq| = |q_bragg_corr - q_pixel_recalc| (Å⁻¹)')
            plt.title('Spatial Distribution of q-vector Differences on Detector')
            plt.xlabel('Fast Pixel Coordinate (px_idx)')
            plt.ylabel('Slow Pixel Coordinate (py_idx)')
            plt.xlim(0, num_fast_plot)
            plt.ylim(num_slow_plot, 0) # Invert Y-axis to match typical image display (origin top-left)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("q_difference_heatmap.png")
            print("\nSaved q_difference_heatmap.png")
            # plt.show()

    except ImportError:
        print("\nMatplotlib not found. Skipping plot generation.")
else:
    print("\nNo reflections were processed for comparison.")
