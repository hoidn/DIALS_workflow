from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family import flex  # For loading reflection table
import numpy as np
from scitbx import matrix  # For matrix/vector operations with crystal model
import math
import os

# --- Configuration ---
INDEXED_EXPT_FILE = "indexed.expt"
INDEXED_REFL_FILE = "indexed.refl"
QMAP_NPY_PREFIX = "panel_"  # Should match what's used in calculate_q_per_pixel.py
# ---------------------

# --- Helper function to get q_bragg ---
def get_q_bragg_from_reflection(refl, experiment):
    """
    Return q_bragg in the DIALS laboratory frame (Å⁻¹) using the
    crystal model's hkl_to_reciprocal_space_vec method.
    This method handles all necessary frame transformations.
    """
    if 'miller_index' not in refl:
        # This check should ideally be done by the caller or handled if refl comes from a source
        # where miller_index might be missing for certain entries.
        # print("Warning: 'miller_index' not found in reflection. Cannot calculate q_bragg.")
        return None

    hkl_tuple = refl["miller_index"]
    try:
        # This is the preferred DIALS method, returns vector in Å⁻¹ in lab frame
        q_bragg_scitbx_vec = experiment.crystal.hkl_to_reciprocal_space_vec(hkl_tuple)
        q_bragg_np = np.array(q_bragg_scitbx_vec.elems)
        return q_bragg_np
    except AttributeError as e:
        # print(f"Warning: experiment.crystal.hkl_to_reciprocal_space_vec not available ({e}). This method is expected.")
        # According to review, if this method is standard, no fallback is needed.
        return None # Or raise error if this method is essential and expected to always exist
    except Exception as e:
        # print(f"Error calculating q_bragg for hkl {hkl_tuple}: {e}")
        return None

# --- Helper function to calculate q_pixel for a specific pixel ---
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
    if D_scattered_norm < 1e-9: return np.array([0.0, 0.0, 0.0])  # Should not happen for Bragg peak
    s1_lab = D_scattered / D_scattered_norm
    k_out = s1_lab * k_magnitude
    
    # 4. q
    q_pixel = k_out - k_in
    return q_pixel

# --- Helper function to read q-vector from precomputed q-maps ---
def get_q_from_maps(panel_id, px_fast_idx, py_slow_idx):
    """Get q-vector for a given pixel from the precomputed q-maps."""
    # Load the q-maps for this panel
    try:
        qx_map = np.load(f"{QMAP_NPY_PREFIX}{panel_id}_qmap_qx.npy")
        qy_map = np.load(f"{QMAP_NPY_PREFIX}{panel_id}_qmap_qy.npy")
        qz_map = np.load(f"{QMAP_NPY_PREFIX}{panel_id}_qmap_qz.npy")
        
        # Get q components at the specified pixel
        # NumPy array indexing is [row, column] = [slow, fast]
        qx = qx_map[py_slow_idx, px_fast_idx]
        qy = qy_map[py_slow_idx, px_fast_idx]
        qz = qz_map[py_slow_idx, px_fast_idx]
        
        return np.array([qx, qy, qz])
    except FileNotFoundError:
        print(f"Warning: Q-maps for panel {panel_id} not found.")
        return None
    except IndexError:
        print(f"Warning: Pixel {px_fast_idx},{py_slow_idx} is outside the q-map boundaries for panel {panel_id}.")
        return None

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

# Check if q-maps exist
qmap_exists = False
for panel_id in range(len(experiments[0].detector)):
    if os.path.exists(f"{QMAP_NPY_PREFIX}{panel_id}_qmap_qx.npy"):
        qmap_exists = True
        break

if not qmap_exists:
    print("Warning: No q-maps found. Please run calculate_q_per_pixel.py first.")
    print("Continuing with direct calculation method only...")

print("\nComparing q_bragg with q_pixel for indexed reflections:")
q_direct_diff_magnitudes = []
q_map_diff_magnitudes = []
q_bragg_mags_list = []
q_pixel_direct_mags_list = []
q_pixel_map_mags_list = []
all_px_coords = []
all_py_coords = []

for i in range(len(reflections_indexed)):
    refl = reflections_indexed[i]
    
    # Get experiment corresponding to this reflection's experiment ID
    experiment_id = refl['id']
    if experiment_id < 0 or experiment_id >= len(experiments):
        print(f"Warning: Invalid experiment ID {experiment_id} for reflection {i}. Skipping.")
        continue
    current_experiment = experiments[experiment_id]

    # a. Calculate q_bragg
    q_bragg = get_q_bragg_from_reflection(refl, current_experiment)  # (3,) NumPy array

    # b. Get pixel coordinates and panel ID for this Bragg peak
    panel_id = refl['panel']
    
    # Get refined pixel coordinates
    if 'xyzcal.mm' in refl:
        x_cal_mm, y_cal_mm, _ = refl['xyzcal.mm']
        
        # Get panel model
        beam_model_current = current_experiment.beam
        detector_model_current = current_experiment.detector
        if panel_id >= len(detector_model_current):
            print(f"  Warning: Panel ID {panel_id} out of range for detector. Skipping reflection {i}.")
            continue
        panel_model_current = detector_model_current[panel_id]
        
        try:
            fast_px_cal, slow_px_cal = panel_model_current.millimeter_to_pixel((x_cal_mm, y_cal_mm))
            px_idx = int(round(fast_px_cal))
            py_idx = int(round(slow_px_cal))
        except RuntimeError as e:
            print(f"  Warning: Could not convert mm to pixel for reflection {i}: {e}")
            # Fallback to observed pixels
            try:
                x_obs_px, y_obs_px, _ = refl['xyzobs.px.value']
                px_idx = int(round(x_obs_px))
                py_idx = int(round(y_obs_px))
                print(f"    Falling back to xyzobs.px.value for pixel coordinates.")
            except KeyError:
                print(f"    xyzobs.px.value also not found. Skipping reflection {i}.")
                continue
    elif 'xyzobs.px.value' in refl:
        x_obs_px, y_obs_px, _ = refl['xyzobs.px.value']
        px_idx = int(round(x_obs_px))
        py_idx = int(round(y_obs_px))
    else:
        print(f"  Warning: No pixel coordinates found for reflection {i}. Skipping.")
        continue

    # c. Calculate q_pixel directly for this specific pixel
    q_pixel_direct = calculate_q_for_single_pixel(
        current_experiment.beam, 
        current_experiment.detector[panel_id],
        px_idx, py_idx
    )
    
    # d. Get q_pixel from precomputed maps if available
    q_pixel_map = get_q_from_maps(panel_id, px_idx, py_idx) if qmap_exists else None
    
    # e. Compare q_bragg with q_pixel_direct
    q_direct_difference = q_bragg - q_pixel_direct
    direct_diff_magnitude = np.linalg.norm(q_direct_difference)
    q_direct_diff_magnitudes.append(direct_diff_magnitude)
    
    # f. Compare q_bragg with q_pixel_map if available
    if q_pixel_map is not None:
        q_map_difference = q_bragg - q_pixel_map
        map_diff_magnitude = np.linalg.norm(q_map_difference)
        q_map_diff_magnitudes.append(map_diff_magnitude)
    
    # Store data for statistics and plotting
    all_px_coords.append(px_idx)
    all_py_coords.append(py_idx)
    q_bragg_mag = np.linalg.norm(q_bragg)
    q_pixel_direct_mag = np.linalg.norm(q_pixel_direct)
    q_bragg_mags_list.append(q_bragg_mag)
    q_pixel_direct_mags_list.append(q_pixel_direct_mag)
    
    if q_pixel_map is not None:
        q_pixel_map_mag = np.linalg.norm(q_pixel_map)
        q_pixel_map_mags_list.append(q_pixel_map_mag)
    
    # Print detailed results for first few reflections and any with large differences
    if i < 10 or direct_diff_magnitude > 0.01:
        print(f"Refl {i} (hkl: {refl['miller_index']}):")
        print(f"  Panel: {panel_id}, Pixel (fast,slow): ({px_idx},{py_idx})")
        print(f"  q_bragg (from A matrix)     : ({q_bragg[0]:.4f}, {q_bragg[1]:.4f}, {q_bragg[2]:.4f}), |q|={q_bragg_mag:.4f}")
        print(f"  q_pixel (direct calculation): ({q_pixel_direct[0]:.4f}, {q_pixel_direct[1]:.4f}, {q_pixel_direct[2]:.4f}), |q|={q_pixel_direct_mag:.4f}")
        print(f"  |q_bragg - q_pixel_direct|  : {direct_diff_magnitude:.6f} Å⁻¹")
        if q_bragg_mag > 1e-6:
            print(f"  Relative diff |q_bragg - q_pixel_direct| / |q_bragg| : {direct_diff_magnitude/q_bragg_mag:.6f}")
        
        if q_pixel_map is not None:
            print(f"  q_pixel (from q-maps)      : ({q_pixel_map[0]:.4f}, {q_pixel_map[1]:.4f}, {q_pixel_map[2]:.4f}), |q|={q_pixel_map_mag:.4f}")
            print(f"  |q_bragg - q_pixel_map|    : {map_diff_magnitude:.6f} Å⁻¹")
            if q_bragg_mag > 1e-6:
                print(f"  Relative diff |q_bragg - q_pixel_map| / |q_bragg| : {map_diff_magnitude/q_bragg_mag:.6f}")

# Print summary statistics
if q_direct_diff_magnitudes:
    q_direct_diff_magnitudes = np.array(q_direct_diff_magnitudes)
    print("\nSummary of |q_bragg - q_pixel_direct| (Å⁻¹):")
    print(f"  Mean:   {np.mean(q_direct_diff_magnitudes):.6f}")
    print(f"  Median: {np.median(q_direct_diff_magnitudes):.6f}")
    print(f"  StdDev: {np.std(q_direct_diff_magnitudes):.6f}")
    print(f"  Min:    {np.min(q_direct_diff_magnitudes):.6f}")
    print(f"  Max:    {np.max(q_direct_diff_magnitudes):.6f}")

if q_map_diff_magnitudes:
    q_map_diff_magnitudes = np.array(q_map_diff_magnitudes)
    print("\nSummary of |q_bragg - q_pixel_map| (Å⁻¹):")
    print(f"  Mean:   {np.mean(q_map_diff_magnitudes):.6f}")
    print(f"  Median: {np.median(q_map_diff_magnitudes):.6f}")
    print(f"  StdDev: {np.std(q_map_diff_magnitudes):.6f}")
    print(f"  Min:    {np.min(q_map_diff_magnitudes):.6f}")
    print(f"  Max:    {np.max(q_map_diff_magnitudes):.6f}")

# Optional: Create plots
try:
    import matplotlib.pyplot as plt
    
    # Set up the figure for plots
    num_plots = 1 + (1 if q_map_diff_magnitudes else 0)
    fig = plt.figure(figsize=(6*num_plots, 10))
    
    # Plot histograms of q-vector differences
    plt.subplot(2, num_plots, 1)
    plt.hist(q_direct_diff_magnitudes, bins=50)
    plt.xlabel("|q_bragg - q_pixel_direct| (Å⁻¹)")
    plt.ylabel("Frequency")
    plt.title("Distribution of q-vector Differences\n(Direct Calculation)")
    plt.grid(True)
    
    if q_map_diff_magnitudes:
        plt.subplot(2, num_plots, 2)
        plt.hist(q_map_diff_magnitudes, bins=50)
        plt.xlabel("|q_bragg - q_pixel_map| (Å⁻¹)")
        plt.ylabel("Frequency")
        plt.title("Distribution of q-vector Differences\n(From Q-Maps)")
        plt.grid(True)
    
    # Plot comparison of q-vector magnitudes
    plt.subplot(2, 1, 2)
    plt.scatter(q_bragg_mags_list, q_pixel_direct_mags_list, alpha=0.5, s=10, label="Direct Calculation")
    
    if q_pixel_map_mags_list:
        plt.scatter(q_bragg_mags_list, q_pixel_map_mags_list, alpha=0.5, s=10, color='red', label="From Q-Maps")
    
    min_val = min(min(q_bragg_mags_list), min(q_pixel_direct_mags_list))
    max_val = max(max(q_bragg_mags_list), max(q_pixel_direct_mags_list))
    
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Ideal Match")  # y=x line
    plt.xlabel("|q_bragg| (Å⁻¹)")
    plt.ylabel("|q_pixel| (Å⁻¹)")
    plt.title("Comparison of q-vector Magnitudes")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("q_consistency_check.png")
    print("\nSaved q_consistency_check.png")
    
    # Create error heatmap
    if all_px_coords and all_py_coords:
        all_px_coords_np = np.array(all_px_coords)
        all_py_coords_np = np.array(all_py_coords)
        
        # Set up figure
        plt.figure(figsize=(12, 10))
        
        # Plot error heatmap for direct calculation
        # Normalize error for color mapping, cap at 95th percentile for better visualization
        vmax_error = np.percentile(q_direct_diff_magnitudes, 95)
        
        scatter = plt.scatter(
            all_px_coords_np, 
            all_py_coords_np, 
            c=q_direct_diff_magnitudes, 
            cmap='viridis',
            s=10,
            vmin=0, 
            vmax=vmax_error
        )
        
        plt.colorbar(scatter, label='|Δq| = |q_bragg - q_pixel| (Å⁻¹)')
        plt.title('Spatial Distribution of q-vector Differences on Detector')
        plt.xlabel('Fast Pixel Coordinate (px_idx)')
        plt.ylabel('Slow Pixel Coordinate (py_idx)')
        
        # Try to get detector dimensions for plot limits
        try:
            panel0 = experiments[0].detector[0]
            fast_dim, slow_dim = panel0.get_image_size()
            plt.xlim(0, fast_dim)
            plt.ylim(slow_dim, 0)  # Invert Y-axis to match image display (origin top-left)
        except Exception:
            # Fallback to using the range of coordinates
            plt.xlim(0, np.max(all_px_coords_np) + 100)
            plt.ylim(np.max(all_py_coords_np) + 100, 0)
        
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("q_difference_heatmap.png")
        print("Saved q_difference_heatmap.png")
except ImportError:
    print("\nMatplotlib not found. Skipping plot generation.")
except Exception as e:
    print(f"\nError during plot generation: {e}")

print("\nQ-vector consistency check completed.")