import os
import sys
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family import flex # For loading reflection table
import numpy as np
from scitbx import matrix # For matrix/vector operations with crystal model
import math

# --- Configuration ---
# These will be determined by the wrapper script or command-line arguments in a real scenario
# For now, using placeholders that should be checked/set by the calling environment.
INDEXED_EXPT_FILE = "indexed_refined_detector.expt" # Default, should be checked by caller
INDEXED_REFL_FILE = "indexed_refined_detector.refl" # Default, should be checked by caller
# ---------------------

# --- Helper function to convert hkl to q in lab frame (dxtbx version independent) ---
def hkl_to_lab_q(experiment, hkl_vec):
    """
    Convert an hkl column vector → reciprocal-space vector in the DIALS
    laboratory frame, for all dxtbx versions.
    """
    A = matrix.sqr(experiment.crystal.get_A())
    S = matrix.sqr(experiment.goniometer.get_setting_rotation())
    F = matrix.sqr(experiment.goniometer.get_fixed_rotation())
    C = matrix.sqr((1,0,0, 0,0,-1, 0,1,0))
    R_lab = C * S * F
    return R_lab * A * hkl_vec

# --- Helper function to get q_bragg ---
def get_q_bragg_from_reflection(refl, experiment):
    """
    Return q_bragg in the DIALS laboratory frame (Å⁻¹) using the
    crystal model's hkl_to_reciprocal_space_vec method or manual calculation.
    """
    try:
        hkl_tuple = refl["miller_index"]
    except KeyError:
        print("Warning: 'miller_index' column not found in reflection. Cannot calculate q_bragg.")
        return None
    
    # Option 1: Use the built-in method if available
    try:
        q_vec_scitbx = experiment.crystal.hkl_to_reciprocal_space_vec(hkl_tuple)
        return np.array(q_vec_scitbx.elems)
    except AttributeError:
        # Option 2: Fall back to manual calculation if needed
        hkl_vec = matrix.col(hkl_tuple) 
        q_bragg_lab_scitbx = hkl_to_lab_q(experiment, hkl_vec)
        return np.array(q_bragg_lab_scitbx.elems)

# --- Helper function to calculate q_pixel for a specific pixel (like in pixq.py) ---
def calculate_q_for_single_pixel(beam_model, panel_model, px_fast_idx, py_slow_idx):
    """Calculates q-vector for a specific pixel directly using beam and detector models."""
    s0 = np.array(beam_model.get_s0())
    k_in = s0
    k_magnitude = np.linalg.norm(s0)
    P_lab = np.array(panel_model.get_pixel_lab_coord((px_fast_idx, py_slow_idx)))
    sample_origin = np.array([0.0, 0.0, 0.0])
    D_scattered = P_lab - sample_origin
    D_scattered_norm = np.linalg.norm(D_scattered)
    if D_scattered_norm < 1e-9: return np.array([0.0, 0.0, 0.0])
    s1_lab = D_scattered / D_scattered_norm
    k_out = s1_lab * k_magnitude
    q_pixel = k_out - k_in
    return q_pixel

# --- Main part ---
def run_consistency_check(expt_file_path, refl_file_path, verbose=False):
    print(f"Loading indexed experiments from: {expt_file_path}")
    if not os.path.exists(expt_file_path):
        print(f"Error: Experiment file not found at {expt_file_path}")
        return
    try:
        experiments = ExperimentListFactory.from_json_file(expt_file_path)
    except Exception as e:
        print(f"Error loading experiment file {expt_file_path}: {e}")
        return

    if len(experiments) == 0:
        print(f"Warning: No experiments found in {expt_file_path}. Cannot perform consistency check.")
        return

    print(f"Loading indexed reflections from: {refl_file_path}")
    if not os.path.exists(refl_file_path):
        print(f"Error: Reflection file not found at {refl_file_path}")
        return
    try:
        reflections = flex.reflection_table.from_file(refl_file_path)
    except Exception as e:
        print(f"Error loading reflection file {refl_file_path}: {e}")
        return

    # Check for essential columns
    essential_cols = ['miller_index', 'id', 'panel', 'xyzcal.mm', 's1']
    missing_cols = [col for col in essential_cols if col not in reflections]
    if missing_cols:
        print(f"Error: Reflection table {refl_file_path} is missing essential columns: {', '.join(missing_cols)}. Cannot perform consistency check.")
        return

    if 'flags' in reflections:
        indexed_sel = reflections.get_flags(reflections.flags.indexed)
        reflections_indexed = reflections.select(indexed_sel)
        print(f"Found {len(reflections_indexed)} indexed reflections (out of {len(reflections)} total).")
    else:
        print("Warning: 'flags' column not in reflection table. Assuming all reflections are indexed.")
        reflections_indexed = reflections

    if len(reflections_indexed) == 0:
        print("No indexed reflections found. Cannot perform consistency check.")
        return

    print("\nComparing q_bragg with directly recalculated q_pixel for indexed reflections:")
    q_diff_magnitudes = []
    q_bragg_mags_list = []
    q_pixel_mags_list = []
    all_px_coords = []
    all_py_coords = []
    q_pixel_vs_dials_pred_diff_norms = flex.double()

    processed_reflections = 0
    for i in range(len(reflections_indexed)):
        refl = reflections_indexed[i]
        
        try:
            experiment_id = refl['id']
            if not isinstance(experiment_id, int):
                experiment_id = int(round(experiment_id))
            if not (0 <= experiment_id < len(experiments)):
                if verbose: print(f"Warning: Invalid experiment ID {experiment_id} for reflection {i}. Skipping.")
                continue
            current_experiment = experiments[experiment_id]
        except (KeyError, ValueError, TypeError) as e:
            if verbose: print(f"Warning: Error accessing experiment ID for reflection {i}: {e}. Skipping.")
            continue

        q_bragg = get_q_bragg_from_reflection(refl, current_experiment)
        if q_bragg is None:
            if verbose: print(f"Warning: Could not calculate q_bragg for reflection {i}. Skipping.")
            continue

        panel_id = refl['panel']
        x_cal_mm, y_cal_mm, _ = refl['xyzcal.mm'] 
        
        beam_model_current = current_experiment.beam
        detector_model_current = current_experiment.detector
        if not (0 <= panel_id < len(detector_model_current)):
            if verbose: print(f"  Warning: Panel ID {panel_id} out of range for detector. Skipping reflection {i}.")
            continue
        panel_model_current = detector_model_current[panel_id]

        try:
            fast_px_cal, slow_px_cal = panel_model_current.millimeter_to_pixel((x_cal_mm, y_cal_mm))
        except RuntimeError as e:
            if verbose: print(f"  Warning: Could not convert mm to pixel for reflection {i} (panel {panel_id}): {e}")
            if 'xyzobs.px.value' in refl:
                if verbose: print("    Falling back to xyzobs.px.value for q_pixel calculation.")
                x_obs_px, y_obs_px, _ = refl['xyzobs.px.value']
                px_idx = int(round(x_obs_px))
                py_idx = int(round(y_obs_px))
            else:
                if verbose: print(f"    xyzobs.px.value also not found. Skipping reflection {i}.")
                continue
        else:
            px_idx = int(round(fast_px_cal))
            py_idx = int(round(slow_px_cal))
        
        q_pixel_recalculated = calculate_q_for_single_pixel(beam_model_current, panel_model_current, px_idx, py_idx)

        if i == 0 and verbose: # Detailed print for the first reflection
            print(f"\n--- Detailed Debug for Refl {i} ---")
            hkl_tuple = refl['miller_index']
            print(f"  Miller Index (hkl): {hkl_tuple}")
            A_matrix_sqr = matrix.sqr(current_experiment.crystal.get_A())
            print(f"  A-matrix elements: {A_matrix_sqr.elems}")
            print(f"  q_bragg (from A matrix): {q_bragg.tolist()}")
            print(f"  Pixel (fast,slow) used for q_pixel_recalc: ({px_idx},{py_idx})")
            print(f"  q_pixel_recalculated (for comparison): {q_pixel_recalculated.tolist()}")
            print(f"--- End Detailed Debug for Refl {i} ---\n")

        q_difference = q_bragg - q_pixel_recalculated
        diff_magnitude = np.linalg.norm(q_difference)
        q_diff_magnitudes.append(diff_magnitude)
        all_px_coords.append(px_idx)
        all_py_coords.append(py_idx)

        q_bragg_mag = np.linalg.norm(q_bragg)
        q_pixel_mag = np.linalg.norm(q_pixel_recalculated)
        q_bragg_mags_list.append(q_bragg_mag)
        q_pixel_mags_list.append(q_pixel_mag)

        s0_vec = np.array(current_experiment.beam.get_s0())
        s1_vec = np.array(refl['s1'])
        q_pred_dials = s1_vec - s0_vec
        
        diff_q_pixel_vs_dials_pred = q_pred_dials - q_pixel_recalculated
        q_pixel_vs_dials_pred_diff_norms.append(np.linalg.norm(diff_q_pixel_vs_dials_pred))
        processed_reflections += 1

        if verbose and (i < 10 or diff_magnitude > 0.01) : # Print first few and any large differences
            print(f"Refl {i} (hkl: {refl['miller_index']}):")
            print(f"  Panel: {panel_id}, Pixel (fast,slow): ({px_idx},{py_idx})")
            print(f"  q_bragg (from A matrix) : ({q_bragg[0]:.4f}, {q_bragg[1]:.4f}, {q_bragg[2]:.4f}), |q|={q_bragg_mag:.4f}")
            print(f"  q_pixel (recalculated)  : ({q_pixel_recalculated[0]:.4f}, {q_pixel_recalculated[1]:.4f}, {q_pixel_recalculated[2]:.4f}), |q|={q_pixel_mag:.4f}")
            print(f"  Difference vector       : ({q_difference[0]:.4f}, {q_difference[1]:.4f}, {q_difference[2]:.4f})")
            print(f"  |q_bragg - q_pixel_recalc|: {diff_magnitude:.6f} Å⁻¹")
            if q_bragg_mag > 1e-6 :
                 print(f"  Relative diff |q_bragg - q_pixel_recalc| / |q_bragg| : {diff_magnitude/q_bragg_mag:.6f}")

    if processed_reflections == 0:
        print("No reflections were successfully processed for q-vector comparison.")
        return

    if q_diff_magnitudes:
        q_diff_magnitudes_np = np.array(q_diff_magnitudes)
        print("\nSummary of |q_bragg - q_pixel_recalculated| (Å⁻¹):")
        print(f"  Processed reflections for summary: {processed_reflections}")
        print(f"  Mean:   {np.mean(q_diff_magnitudes_np):.6f}")
        print(f"  Median: {np.median(q_diff_magnitudes_np):.6f}")
        print(f"  StdDev: {np.std(q_diff_magnitudes_np):.6f}")
        print(f"  Min:    {np.min(q_diff_magnitudes_np):.6f}")
        print(f"  Max:    {np.max(q_diff_magnitudes_np):.6f}")

        if len(q_pixel_vs_dials_pred_diff_norms) > 0:
            print(f"\nMean |q_pred_dials - q_pixel_recalculated|: {flex.mean(q_pixel_vs_dials_pred_diff_norms):.6f} Å⁻¹")

        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.hist(q_diff_magnitudes_np, bins=50)
            plt.xlabel("|q_bragg - q_pixel_recalculated| (Å⁻¹)")
            plt.ylabel("Frequency")
            plt.title("Distribution of q-vector Differences (Direct Recalculation)")
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.scatter(q_bragg_mags_list, q_pixel_mags_list, alpha=0.5, s=10)
            min_val = min(min(q_bragg_mags_list), min(q_pixel_mags_list)) if q_bragg_mags_list and q_pixel_mags_list else 0
            max_val = max(max(q_bragg_mags_list), max(q_pixel_mags_list)) if q_bragg_mags_list and q_pixel_mags_list else 1
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal Match")
            plt.xlabel("|q_bragg| (Å⁻¹)")
            plt.ylabel("|q_pixel_recalculated| (Å⁻¹)")
            plt.title("Comparison of q-vector Magnitudes")
            plt.legend()
            plt.axis('equal')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig("q_consistency_check_direct_recalc.png")
            print("\nSaved q_consistency_check_direct_recalc.png")

            if all_px_coords and all_py_coords:
                all_px_coords_np = np.array(all_px_coords)
                all_py_coords_np = np.array(all_py_coords)
                
                try:
                    panel0 = experiments[0].detector[0]
                    num_fast_plot, num_slow_plot = panel0.get_image_size()
                except Exception as e:
                    print(f"Warning: Error getting panel dimensions for heatmap: {e}. Using max coordinates.")
                    num_fast_plot = np.max(all_px_coords_np) + 1 if len(all_px_coords_np) > 0 else 2463
                    num_slow_plot = np.max(all_py_coords_np) + 1 if len(all_py_coords_np) > 0 else 2527

                plt.figure(figsize=(10, 10))
                vmax_error = 0.1 # Adjusted fixed cap
                if np.any(q_diff_magnitudes_np > vmax_error):
                     print(f"Note: Some |Δq| values exceed heatmap vmax of {vmax_error} Å⁻¹ and will be shown as the max color.")

                scatter = plt.scatter(
                    all_px_coords_np, 
                    all_py_coords_np, 
                    c=q_diff_magnitudes_np, 
                    cmap='viridis',
                    s=10,
                    vmin=0, 
                    vmax=vmax_error
                )
                
                plt.colorbar(scatter, label='|Δq| = |q_bragg - q_pixel_recalc| (Å⁻¹)')
                plt.title('Spatial Distribution of q-vector Differences on Detector')
                plt.xlabel('Fast Pixel Coordinate (px_idx)')
                plt.ylabel('Slow Pixel Coordinate (py_idx)')
                plt.xlim(0, num_fast_plot)
                plt.ylim(num_slow_plot, 0)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig("q_difference_heatmap.png")
                print("\nSaved q_difference_heatmap.png")

        except ImportError:
            print("\nMatplotlib not found. Skipping plot generation.")
    else:
        print("\nNo reflections were successfully processed for comparison.")

if __name__ == "__main__":
    # This part is for direct execution. 
    # In a real pipeline, these paths might come from args or a wrapper.
    if not os.path.exists(INDEXED_EXPT_FILE):
        print(f"Error: Default experiment file {INDEXED_EXPT_FILE} not found in current directory.")
        print("Please provide experiment and reflection files, or place defaults here.")
        sys.exit(1)
    if not os.path.exists(INDEXED_REFL_FILE):
        print(f"Error: Default reflection file {INDEXED_REFL_FILE} not found in current directory.")
        print("Please provide experiment and reflection files, or place defaults here.")
        sys.exit(1)
        
    run_consistency_check(INDEXED_EXPT_FILE, INDEXED_REFL_FILE, verbose=True)
