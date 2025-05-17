from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family import flex # For loading reflection table
import numpy as np
from scitbx import matrix # For matrix/vector operations with crystal model
import math

# --- Configuration ---
INDEXED_EXPT_FILE = "indexed.expt"
INDEXED_REFL_FILE = "indexed.refl"
# Assuming q-maps are saved with this prefix and panel ID
QMAP_NPY_PREFIX = "panel_"
# ---------------------

# 1. Load indexed experiments and reflections
print(f"Loading indexed experiments from: {INDEXED_EXPT_FILE}")
experiments = ExperimentListFactory.from_json_file(INDEXED_EXPT_FILE)
print(f"Loading indexed reflections from: {INDEXED_REFL_FILE}")
reflections = flex.reflection_table.from_file(INDEXED_REFL_FILE)

# Filter for indexed reflections if necessary (dials.index usually outputs only indexed ones)
indexed_sel = reflections.get_flags(reflections.flags.indexed)
reflections_indexed = reflections.select(indexed_sel)
print(f"Found {len(reflections_indexed)} indexed reflections.")

if len(reflections_indexed) == 0:
    print("No indexed reflections found. Cannot perform consistency check.")
    exit()

# --- Helper function to get q_bragg ---
def get_q_bragg_from_reflection(reflection, experiment):
    """Calculates q-vector from Miller index and crystal model."""
    h, k, l = reflection['miller_index']
    crystal_model = experiment.crystal

    # Get the A matrix (setting matrix A = UB)
    # The A matrix transforms fractional Miller indices (h,k,l) to
    # reciprocal space coordinates (x*, y*, z*) in Å⁻¹.
    # A = [a*x b*x c*x]
    #     [a*y b*y c*y]
    #     [a*z b*z c*z]  <-- This is if A is [a* b* c*] (columns are basis vectors)
    #
    # For dxtbx, r_star = A * h where h is a column vector (h,k,l)^T
    # So, the columns of A are indeed a*, b*, c*
    
    A_matrix = matrix.sqr(crystal_model.get_A()) # A is 3x3 matrix

    # a_star_vec is the first column of A
    a_star_vec = A_matrix.col(0)
    # b_star_vec is the second column of A
    b_star_vec = A_matrix.col(1)
    # c_star_vec is the third column of A
    c_star_vec = A_matrix.col(2)

    # q_bragg = h * a* + k * b* + l * c*
    q_bragg = h * a_star_vec + k * b_star_vec + l * c_star_vec
    
    return np.array(q_bragg) # Convert scitbx.matrix.col to NumPy array (3,)

# --- Load per-pixel q-maps (assuming only one panel for simplicity in this example) ---
# In a real scenario, you'd loop through panels or have a way to map panel_id from reflection
# to the correct q-map files.

# For this example, let's assume all reflections are on panel 0
# and load its q-maps.
# You'll need to adapt this if you have multiple panels.
q_maps_loaded = {}
max_panel_id = 0 # Determine this from your detector or reflection table
if 'panel' in reflections_indexed.keys():
    max_panel_id = flex.max(reflections_indexed['panel'])

for pid in range(max_panel_id + 1):
    try:
        print(f"Loading q-maps for panel {pid}...")
        qx_map = np.load(f"{QMAP_NPY_PREFIX}{pid}_qmap_qx.npy")
        qy_map = np.load(f"{QMAP_NPY_PREFIX}{pid}_qmap_qy.npy")
        qz_map = np.load(f"{QMAP_NPY_PREFIX}{pid}_qmap_qz.npy")
        q_maps_loaded[pid] = {'qx': qx_map, 'qy': qy_map, 'qz': qz_map}
        print(f"  Loaded qx_map shape: {qx_map.shape}")
    except FileNotFoundError:
        print(f"  Warning: q-map files for panel {pid} not found. Skipping.")
        q_maps_loaded[pid] = None


# --- Perform Comparison ---
print("\nComparing q_bragg with q_pixelmap for indexed reflections:")
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
    
    # Ensure pixel coordinates are within bounds of the loaded q-map for that panel
    if q_maps_loaded.get(panel_id) is None:
        # print(f"  Skipping reflection {i} on panel {panel_id} (q-maps not loaded).")
        continue

    q_panel_map_qx = q_maps_loaded[panel_id]['qx']
    num_slow, num_fast = q_panel_map_qx.shape # (num_slow_pixels, num_fast_pixels)

    # Convert observed pixel coordinates (float) to integer indices
    # DIALS coordinates: fast (x), slow (y)
    # Array indices: row (slow), col (fast)
    px_idx = int(round(x_obs_px)) # Fast scan = column index
    py_idx = int(round(y_obs_px)) # Slow scan = row index

    if not (0 <= px_idx < num_fast and 0 <= py_idx < num_slow):
        print(f"  Warning: Pixel ({px_idx}, {py_idx}) for reflection {i} on panel {panel_id} is out of q-map bounds ({num_fast}, {num_slow}). Skipping.")
        continue
        
    # c. Look up q_pixelmap from the pre-calculated maps
    # Remember q_maps are indexed [slow_idx, fast_idx]
    q_pixelmap_x = q_maps_loaded[panel_id]['qx'][py_idx, px_idx]
    q_pixelmap_y = q_maps_loaded[panel_id]['qy'][py_idx, px_idx]
    q_pixelmap_z = q_maps_loaded[panel_id]['qz'][py_idx, px_idx]
    q_pixelmap = np.array([q_pixelmap_x, q_pixelmap_y, q_pixelmap_z]) # (3,) NumPy array

    # d. Compare
    q_difference = q_bragg - q_pixelmap
    diff_magnitude = np.linalg.norm(q_difference)
    q_diff_magnitudes.append(diff_magnitude)

    q_bragg_mag = np.linalg.norm(q_bragg)
    q_pixel_mag = np.linalg.norm(q_pixelmap)
    q_bragg_mags_list.append(q_bragg_mag)
    q_pixel_mags_list.append(q_pixel_mag)

    if i < 10 or diff_magnitude > 0.01 : # Print first few and any large differences
        print(f"Refl {i} (hkl: {refl['miller_index']}):")
        print(f"  Panel: {panel_id}, Pixel (fast,slow): ({px_idx},{py_idx})")
        print(f"  q_bragg     (Å⁻¹): ({q_bragg[0]:.4f}, {q_bragg[1]:.4f}, {q_bragg[2]:.4f}), |q|={q_bragg_mag:.4f}")
        print(f"  q_pixelmap  (Å⁻¹): ({q_pixelmap[0]:.4f}, {q_pixelmap[1]:.4f}, {q_pixelmap[2]:.4f}), |q|={q_pixel_mag:.4f}")
        print(f"  Difference vector: ({q_difference[0]:.4f}, {q_difference[1]:.4f}, {q_difference[2]:.4f})")
        print(f"  |q_bragg - q_pixelmap|: {diff_magnitude:.6f} Å⁻¹")
        if q_bragg_mag > 1e-6 : # Avoid division by zero for central beam
             print(f"  Relative diff |q_bragg - q_pixelmap| / |q_bragg| : {diff_magnitude/q_bragg_mag:.6f}")


if q_diff_magnitudes:
    q_diff_magnitudes = np.array(q_diff_magnitudes)
    print("\nSummary of |q_bragg - q_pixelmap| (Å⁻¹):")
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
        plt.xlabel("|q_bragg - q_pixelmap| (Å⁻¹)")
        plt.ylabel("Frequency")
        plt.title("Distribution of q-vector Differences")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.scatter(q_bragg_mags_list, q_pixel_mags_list, alpha=0.5, s=10)
        min_val = min(min(q_bragg_mags_list), min(q_pixel_mags_list))
        max_val = max(max(q_bragg_mags_list), max(q_pixel_mags_list))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal Match") # y=x line
        plt.xlabel("|q_bragg| (Å⁻¹)")
        plt.ylabel("|q_pixelmap| (Å⁻¹)")
        plt.title("Comparison of q-vector Magnitudes")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("q_consistency_check.png")
        print("\nSaved q_consistency_check.png")
        # plt.show() # Uncomment to display plot interactively
    except ImportError:
        print("\nMatplotlib not found. Skipping plot generation.")
else:
    print("\nNo reflections were processed for comparison.")
