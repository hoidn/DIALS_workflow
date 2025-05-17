from dxtbx.model.experiment_list import ExperimentListFactory
import numpy as np
import os

# --- Configuration ---
INDEXED_EXPT_FILE = "indexed.expt"
QMAP_NPY_PREFIX = "panel_"
# ---------------------

# 1. Load indexed experiments
print(f"Loading indexed experiments from: {INDEXED_EXPT_FILE}")
experiments = ExperimentListFactory.from_json_file(INDEXED_EXPT_FILE)
experiment = experiments[0]  # Assuming single experiment

# Get beam and detector
beam = experiment.beam
detector = experiment.detector

# Debug beam direction
s0_tuple = beam.get_s0()
print(f"DEBUG: beam.get_s0() from dxtbx: {s0_tuple}")
s0_vec = np.array(s0_tuple)
print(f"DEBUG: s0_vec magnitude: {np.linalg.norm(s0_vec)}")

# Calculate k_in (incident wavevector)
wavelength = beam.get_wavelength()
k_magnitude = 1.0 / wavelength  # |k| = 1/λ in Å⁻¹
print(f"DEBUG: Wavelength: {wavelength} Å")
print(f"DEBUG: k_magnitude: {k_magnitude} Å⁻¹")

# Normalize s0 to get unit vector in beam direction
s0_unit = s0_vec / np.linalg.norm(s0_vec)
print(f"DEBUG: s0_unit: {s0_unit}")

# k_in = -|k| * s0_unit (negative because s0 points from sample to source)
k_in = -k_magnitude * s0_unit
print(f"DEBUG: k_in: {k_in}")

# Sample interaction point (usually at origin in lab frame)
sample_interaction_point = np.array([0.0, 0.0, 0.0])

# Process each panel
for panel_id, panel in enumerate(detector):
    # Debug panel geometry for panel 0
    if panel_id == 0:
        print(f"DEBUG: Panel 0 Origin (mm): {panel.get_origin()}")
        print(f"DEBUG: Panel 0 Fast Axis: {panel.get_fast_axis()}")
        print(f"DEBUG: Panel 0 Slow Axis: {panel.get_slow_axis()}")
        print(f"DEBUG: Panel 0 Pixel Size (mm): {panel.get_pixel_size()}")
        print(f"DEBUG: Panel 0 Image Size (pixels): {panel.get_image_size()}")
    
    # Get panel dimensions
    fast_dim, slow_dim = panel.get_image_size()  # (width, height) in pixels
    
    # Create arrays for pixel coordinates
    # Note: pixel coordinates in DIALS are (column, row) = (fast, slow)
    slow_indices, fast_indices = np.indices((slow_dim, fast_dim))
    
    # Add 0.5 to get pixel centers
    slow_indices = slow_indices.astype(np.float64) + 0.5
    fast_indices = fast_indices.astype(np.float64) + 0.5
    
    # Get panel vectors
    origin = np.array(panel.get_origin())
    fast_axis = np.array(panel.get_fast_axis())
    slow_axis = np.array(panel.get_slow_axis())
    pixel_size = np.array(panel.get_pixel_size())
    
    # Calculate lab coordinates for all pixels (vectorized)
    # P_lab = origin + (fast_idx * pixel_size_fast * fast_axis) + (slow_idx * pixel_size_slow * slow_axis)
    P_lab_all_pixels = (
        origin +
        np.outer(np.ones(slow_dim), fast_indices.flatten()).reshape(slow_dim, fast_dim) * pixel_size[0] * fast_axis.reshape(1, 1, 3) +
        np.outer(slow_indices.flatten(), np.ones(fast_dim)).reshape(slow_dim, fast_dim) * pixel_size[1] * slow_axis.reshape(1, 1, 3)
    )
    
    # Check P_lab for a specific test pixel
    if panel_id == 0:
        test_px_fast = 2342
        test_px_slow = 30  # This is the index for the slow dimension (row)
        P_lab_test_pixel = P_lab_all_pixels[test_px_slow, test_px_fast, :]
        print(f"DEBUG: P_lab for pixel (fast={test_px_fast}, slow={test_px_slow}) on Panel 0: {P_lab_test_pixel}")
        
        # Manual calculation for this one pixel for cross-check:
        manual_P_lab = (np.array(panel.get_origin()) +
                        ((test_px_fast + 0.5) * panel.get_pixel_size()[0] * np.array(panel.get_fast_axis())) +
                        ((test_px_slow + 0.5) * panel.get_pixel_size()[1] * np.array(panel.get_slow_axis())))
        print(f"DEBUG: Manual P_lab for same pixel: {manual_P_lab}")
        print(f"DEBUG: Difference P_lab_vectorized vs manual: {P_lab_test_pixel - manual_P_lab}")

        # Also check the dxtbx direct method
        dxtbx_P_lab = np.array(panel.get_pixel_lab_coord((test_px_fast, test_px_slow)))
        print(f"DEBUG: dxtbx get_pixel_lab_coord for same pixel: {dxtbx_P_lab}")
        print(f"DEBUG: Difference P_lab_vectorized vs dxtbx: {P_lab_test_pixel - dxtbx_P_lab}")
        
        # Check k_out for that test pixel
        D_scattered_test_pixel = P_lab_test_pixel - sample_interaction_point
        s1_lab_test_pixel = D_scattered_test_pixel / np.linalg.norm(D_scattered_test_pixel)
        k_out_test_pixel = s1_lab_test_pixel * k_magnitude
        print(f"DEBUG: D_scattered_test_pixel: {D_scattered_test_pixel}")
        print(f"DEBUG: s1_lab_test_pixel: {s1_lab_test_pixel}")
        print(f"DEBUG: k_out_test_pixel: {k_out_test_pixel}")
        q_test_pixel = k_out_test_pixel - k_in
        print(f"DEBUG: q for test pixel (from pixel map logic): {q_test_pixel}")
    
    # Calculate scattered beam vectors for all pixels
    # D_scattered = P_lab - sample_interaction_point
    D_scattered = P_lab_all_pixels - sample_interaction_point.reshape(1, 1, 3)
    
    # Normalize to get unit vectors (s1)
    # We need to calculate the norm for each pixel's vector
    D_norms = np.sqrt(np.sum(D_scattered**2, axis=2)).reshape(slow_dim, fast_dim, 1)
    s1_lab = D_scattered / D_norms
    
    # Calculate k_out = |k| * s1 for all pixels
    k_out = s1_lab * k_magnitude
    
    # Calculate q = k_out - k_in for all pixels
    q_vectors = k_out - k_in.reshape(1, 1, 3)
    
    # Extract components
    qx_map = q_vectors[:, :, 0]
    qy_map = q_vectors[:, :, 1]
    qz_map = q_vectors[:, :, 2]
    
    # Save q-maps
    print(f"Saving q-maps for panel {panel_id}...")
    np.save(f"{QMAP_NPY_PREFIX}{panel_id}_qmap_qx.npy", qx_map)
    np.save(f"{QMAP_NPY_PREFIX}{panel_id}_qmap_qy.npy", qy_map)
    np.save(f"{QMAP_NPY_PREFIX}{panel_id}_qmap_qz.npy", qz_map)
    print(f"  Saved q-maps with shape: {qx_map.shape}")

print("Done calculating and saving q-maps.")
