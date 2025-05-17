from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family import flex # For loading reflection table
import numpy as np
from scitbx import matrix # For matrix/vector operations with crystal model
import math

# --- Configuration ---
INDEXED_EXPT_FILE = "indexed.expt"
INDEXED_REFL_FILE = "indexed.refl"
# ---------------------

# --- Helper function to get q_bragg ---
def get_q_bragg_from_reflection(reflection, experiment):
    """Return q = h a* + k b* + l c* in the lab frame (Å⁻¹)."""
    h, k, l = reflection['miller_index']
    crystal_model = experiment.crystal

    # A_matrix_elements is the 9-tuple from crystal.get_A(), row-major: (A0 … A8)
    A_matrix_elements = crystal_model.get_A()

    # The rows of A are a*, b*, c*:
    # a* = (A0, A1, A2)
    # b* = (A3, A4, A5)
    # c* = (A6, A7, A8)
    # We slice the tuple to get the components for each reciprocal lattice vector.
    a_star_vec = matrix.col(A_matrix_elements[0:3])  # (A0, A1, A2)
    b_star_vec = matrix.col(A_matrix_elements[3:6])  # (A3, A4, A5)
    c_star_vec = matrix.col(A_matrix_elements[6:9])  # (A6, A7, A8)

    # q_bragg = h * a* + k * b* + l * c*
    q_bragg_scitbx = h * a_star_vec + k * b_star_vec + l * c_star_vec
    
    return np.array(q_bragg_scitbx) # (3,) NumPy array

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
    x_obs_px, y_obs_px, _ = refl['xyzobs.px.value']
    
    # Convert observed pixel coordinates (float) to integer indices
    # DIALS coordinates: fast (x), slow (y)
    px_idx = int(round(x_obs_px)) # Fast scan = column index
    py_idx = int(round(y_obs_px)) # Slow scan = row index

    # c. Recalculate q_pixel for this specific pixel using the current experiment's models
    beam_model_current = current_experiment.beam
    detector_model_current = current_experiment.detector
    if panel_id >= len(detector_model_current):
        print(f"  Warning: Panel ID {panel_id} out of range for detector. Skipping reflection {i}.")
        continue
    panel_model_current = detector_model_current[panel_id]
    
    q_pixel_recalculated = calculate_q_for_single_pixel(beam_model_current, panel_model_current, px_idx, py_idx)

    # d. Compare
    q_difference = q_bragg - q_pixel_recalculated
    diff_magnitude = np.linalg.norm(q_difference)
    q_diff_magnitudes.append(diff_magnitude)

    q_bragg_mag = np.linalg.norm(q_bragg)
    q_pixel_mag = np.linalg.norm(q_pixel_recalculated)
    q_bragg_mags_list.append(q_bragg_mag)
    q_pixel_mags_list.append(q_pixel_mag)

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
    except ImportError:
        print("\nMatplotlib not found. Skipping plot generation.")
else:
    print("\nNo reflections were processed for comparison.")
