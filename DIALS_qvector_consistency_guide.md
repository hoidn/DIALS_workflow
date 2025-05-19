## Document: Achieving q-vector Consistency in DIALS Stills Processing

**Version:** 1.0
**Date:** May 16, 2024
**Project:** q-space Analysis from DIALS Processed Stills
**Author(s)/Investigator(s):** Ollie (primary), AI Assistant
**Associated Files:** `consistency.py`, `pixq.py`, `process_cbfs_dials.sh`, `*.phil` files, `lys_nitr_8_2_0110.cbf` (example data)

**Table of Contents:**

1.  **Executive Summary**
2.  **Introduction: The q-vector Consistency Challenge**
    *   Motivation
    *   Definition of q-vectors being compared:
        *   `q_bragg`: Derived from the crystal model (A-matrix and HKLs).
        *   `q_pixel`: Derived from experimental geometry (beam, detector, pixel position).
3.  **Initial Problematic Observations & Symptoms**
    *   Large `|Δq| = |q_bragg - q_pixel|` residuals (mean ~0.4-0.5 Å⁻¹).
    *   Systematic spatial error pattern on the detector (heatmap).
    *   Initial attempts and empirical fixes (e.g., Y-axis flip).
4.  **Chronological Debugging Journey & Key Breakthroughs**
    *   Phase 1: Validating `q_pixel` Calculation (`pixq.py` and `consistency.py`)
    *   Phase 2: Investigating `q_bragg` Calculation and the A-matrix
    *   Phase 3: The Role of Detector Geometry Refinement
    *   Phase 4: The Goniometer Frame and Laboratory Frame Mismatch (The "Aha!" Moment)
5.  **Root Cause Analysis: The Full Transformation Chain**
    *   The DIALS Laboratory Frame
    *   The Goniometer Frame and `crystal.get_A()`
    *   The Missing Link: `S`, `F`, and the Constant Rotation `C`
6.  **Implemented Solution & Code Modifications**
    *   Corrected `get_q_bragg_from_reflection` in `consistency.py`
    *   Using `experiment.crystal.hkl_to_reciprocal_space_vec()` (Recommended)
    *   Importance of using refined spot coordinates (`xyzcal.mm` or `xyzcal.px.value`)
    *   Refinement Strategy in DIALS (`dials.index` vs. `dials.refine`)
7.  **Results After Fixes**
    *   Significant reduction in `|Δq|` residuals.
    *   Disappearance of systematic error patterns in heatmaps.
    *   Confirmation of consistency: `q_bragg_lab ≈ q_pixel ≈ (s1 - s0)_from_table`.
8.  **Key Learnings & Best Practices**
    *   Reference Frames are Paramount.
    *   Static Goniometer Rotations Matter for Stills.
    *   Ad-hoc Fixes are Indicators of Deeper Issues.
    *   Leverage DIALS Convenience Functions.
    *   Iterative Refinement and Diagnostics are Essential.
    *   Importance of Comprehensive Model Refinement (Detector, Beam, Crystal).
9.  **Developer Guide: Ensuring q-vector Consistency**
    *   Calculating `q_bragg` in the Lab Frame.
    *   Calculating `q_pixel` from Geometry.
    *   Recommended DIALS Refinement Strategies for Stills.
    *   Verification Steps.
10. **Project User Reference: Interpreting `consistency.py` Outputs**
    *   Key statistics (`|Δq|` mean, median, max).
    *   Interpreting heatmaps and component plots.
    *   Troubleshooting common issues.
11. **Future Work & Recommendations**
    *   DIALS Versioning.
    *   Automated Testing.
    *   Further exploration of subtle residuals.
12. **Appendix: Example Script Snippets (Final Corrected Versions)**
    *   `get_q_bragg_from_reflection` (final version)
    *   Relevant part of `consistency.py` for `q_pixel`
    *   Example `dials.refine` PHIL file for stills.

---

### 1. Executive Summary

This document details the investigation and resolution of significant discrepancies observed between two methods of calculating scattering vectors (q-vectors) from DIALS-processed still diffraction data. Initially, `q_bragg` (derived from the crystal's A-matrix and Miller indices) and `q_pixel` (derived from the refined beam/detector geometry and observed pixel positions) showed a mean difference `|Δq|` of ~0.4-0.5 Å⁻¹ with a strong systematic spatial error pattern.

The root cause was identified as a coordinate frame mismatch: `crystal.get_A()` returns the crystal setting matrix in a goniometer-based frame, while `q_pixel` is calculated in the DIALS laboratory frame. The full transformation requires applying goniometer fixed (`F`) and setting (`S`) rotations, and critically for this setup, a constant rotation (`C`) to map the goniometer frame to the laboratory frame.

The solution involved implementing the complete transformation `q_lab = C * S * F * A_gonio * hkl` for `q_bragg`, or preferably, using the DIALS convenience function `experiment.crystal.hkl_to_reciprocal_space_vec()`. Additionally, ensuring comprehensive refinement of detector, beam, and crystal orientation parameters using `dials.refine` (or appropriately configured `dials.index`) was crucial. Using refined spot coordinates (`xyzcal.mm`) for `q_pixel` calculation further improved precision.

After these fixes, the `|Δq|` residuals are expected to drop to the numerical noise floor (e.g., mean < 0.01 Å⁻¹), and systematic error patterns in spatial heatmaps should be eliminated, confirming the self-consistency of the refined DIALS models.

### 2. Introduction: The q-vector Consistency Challenge

**Motivation:**
Accurate determination of scattering vectors (q-vectors) is fundamental for numerous crystallographic analyses, including diffuse scattering mapping, precise peak analysis, and understanding reciprocal space. Discrepancies between q-vectors derived from different aspects of a processed crystallographic model can indicate errors in data processing, model refinement, or interpretation of coordinate systems. This investigation aimed to achieve high consistency between q-vectors derived from the crystal model and those from the experimental geometry for still diffraction images processed with DIALS.

**Definition of q-vectors being compared:**

*   **`q_bragg`**: This vector represents the theoretical position of a reciprocal lattice point (RLP) for a given Miller index (h,k,l). It is traditionally calculated using the crystal's A-matrix (which combines unit cell parameters and crystal orientation).
    *   Initial approach: `q_bragg ≈ A_crystal @ hkl`
*   **`q_pixel`**: This vector represents the scattering vector corresponding to a specific pixel on the detector where a Bragg reflection is observed (or predicted). It is calculated from the incident beam vector (`k_in`), the scattered beam vector (`k_out` pointing from sample to pixel), and the experimental geometry (beam source, detector panel position, and orientation).
    *   Calculation: `q_pixel = k_out - k_in` (where `|k_out| = |k_in| = 1/λ` for elastic scattering).

### 3. Initial Problematic Observations & Symptoms

The initial runs of `consistency.py` revealed significant issues:

*   **Large `|Δq|` Residuals:** The mean magnitude of the difference vector, `|q_bragg - q_pixel|`, was consistently around 0.4-0.5 Å⁻¹, with maximum values exceeding 1.0 Å⁻¹. This indicated a substantial disagreement.
*   **Systematic Spatial Error Pattern:** The heatmap plotting `|Δq|` against detector pixel coordinates showed a clear, non-random pattern. Typically, errors were smallest at the top of the detector (`py_idx` small) and increased systematically towards the bottom.
*   **Empirical Y-axis Flip:** An ad-hoc multiplication of the Y-component of `q_bragg` by -1 (`q_bragg_np[1] *= -1`) was found to reduce the overall `|Δq|` and make the X and Z components align better in sign, suggesting a Y-axis convention mismatch. However, this did not eliminate the systematic error pattern or reduce the mean `|Δq|` to an acceptable level.
*   **`q_pixel` vs. DIALS `q_pred`:** The `q_pixel` calculated by `consistency.py` (and `pixq.py`) showed excellent agreement (mean difference `< 0.001 Å⁻¹`) with `q_pred = s1 - s0` where `s1` was taken from the DIALS reflection table. This validated the `q_pixel` calculation method itself, suggesting the issue lay with `q_bragg` or its comparison to `q_pixel`.

### 4. Chronological Debugging Journey & Key Breakthroughs

The investigation proceeded through several phases:

*   **Phase 1: Validating `q_pixel` Calculation:**
    *   Confirmed internal consistency of `pixq.py`'s q-map generation with `dxtbx.model.Panel.get_pixel_lab_coord()`.
    *   Confirmed `q_pixel_recalc` in `consistency.py` matched `q_pred_dials` (from `refl['s1'] - beam.get_s0()`) when using the same `experiments.json`, indicating the geometric calculation of `q_pixel` was correct based on the provided model.
    *   Ensured all q-vectors used consistent units (Å⁻¹, `1/λ` scaling, not `2π/λ`).

*   **Phase 2: Investigating `q_bragg` Calculation and the A-matrix:**
    *   Correctly interpreted `crystal.get_A()` as providing elements of the A-matrix, and `A@hkl` as the method to get `q_bragg` components.
    *   The empirical Y-flip was introduced here to improve component-wise agreement, hinting at a coordinate system issue.

*   **Phase 3: The Role of Detector Geometry Refinement:**
    *   Hypothesis: An unrefined or poorly refined detector model in `indexed.expt` (from `dials.index` with default stills settings) was causing the systematic `|Δq|` pattern.
    *   Attempt 1 (Refinement in `dials.index`): Modified `dials.index` via a PHIL file (`indexing_params.phil`) to enable detector refinement (`detector.fix=None`, `detector.hierarchy_level=0`).
        *   Result: `dials.index` completed. The heatmap pattern changed when the Y-flip was OFF (minimums shifted to detector center), but with Y-flip ON, the original top-to-bottom error pattern and `|Δq|` magnitude persisted. This indicated the internal refinement in `dials.index` wasn't fully resolving the geometric inconsistency in the desired way.
    *   Attempt 2 (Dedicated `dials.refine`): Added a separate `dials.refine` step after `dials.index` to explicitly refine detector geometry (`detector.fix=None`, `beam.fix=None`, `crystal.fix=cell`, `scan_varying=False`).
        *   Initial `dials.refine` failure due to incorrect command-line parameters (`origin=shift`).
        *   Corrected `dials.refine` invocation using a PHIL file.
        *   Second `dials.refine` failure due to `scan_varying=True` being default and problematic for stills ("Too few reflections to parameterise...").
        *   Successful `dials.refine` run after setting `scan_varying=False` and ensuring `beam.fix=None` was correctly applied.
        *   The A-matrix from this refined model showed a change in the sign of the Y-component of `A@hkl` (before any script Y-flip), suggesting `dials.refine` was bringing the crystal frame closer to the lab frame convention.

*   **Phase 4: The Goniometer Frame and Laboratory Frame Mismatch (The "Aha!" Moment):**
    *   A separate detailed analysis (external to the live session) proposed that `crystal.get_A()` returns the A-matrix (UB) in the *goniometer frame*.
    *   To transform this to the DIALS *laboratory frame* (where `q_pixel` is defined), one must apply:
        1.  `F`: Goniometer fixed rotation.
        2.  `S`: Goniometer setting rotation.
        3.  `C`: A constant rotation specific to the beamline/setup that maps the goniometer XYZ to the lab XYZ. For this setup, `C` was identified as a -90° rotation about +X (`lab_x = gonio_x`, `lab_y = -gonio_z`, `lab_z = gonio_y`).
    *   This hypothesis elegantly explained why the empirical Y-flip was partially helpful and why simply refining the detector wasn't the complete solution if this `C` transformation was missing.

### 5. Root Cause Analysis: The Full Transformation Chain

The fundamental issue was a mismatch in coordinate reference frames:

*   **DIALS Laboratory Frame:** `q_pixel` is calculated in this frame, typically defined with Z along/opposite the beam, Y vertical, X horizontal. Detector and beam models from `experiments.json` live in this frame.
*   **Goniometer Frame:** `experiment.crystal.get_A()` (the "A-matrix" or "UB matrix") defines the crystal's reciprocal lattice vectors in a Cartesian system attached to the goniometer at its datum.
*   **Transformations:**
    1.  `A_gonio @ hkl` gives `q` in the goniometer frame.
    2.  `F` (fixed rotation) and `S` (setting rotation) are intrinsic goniometer rotations: `q_gonio_rotated = S * F * A_gonio @ hkl`. For stills, `S` and `F` are often identity if no explicit goniometer angles are set, but they are part of the formal chain.
    3.  `C` (constant rotation) transforms from the (potentially rotated) goniometer frame to the DIALS laboratory frame: `q_lab = C @ q_gonio_rotated`.
*   The scripts were initially missing the `C`, `S`, and `F` transformations when calculating `q_bragg` from `A_gonio @ hkl`. The empirical Y-flip was an incomplete attempt to account for components of `C`.

### 6. Implemented Solution & Code Modifications

The final, correct approach involves ensuring both `q_bragg` and `q_pixel` are in the same DIALS laboratory frame.

**1. Corrected `get_q_bragg_from_reflection` in `consistency.py`:**

*   **Preferred Method (using DIALS built-in):**
    ```python
    from scitbx import matrix # At top of file
    import numpy as np      # At top of file

    def get_q_bragg_from_reflection(refl, experiment):
        hkl_tuple = refl["miller_index"]
        # This function handles all necessary crystal and goniometer rotations (C, S, F, A)
        q_vec_scitbx = experiment.crystal.hkl_to_reciprocal_space_vec(hkl_tuple)
        q_bragg_np = np.array(q_vec_scitbx.elems)
        # No manual Y-flip needed here
        return q_bragg_np
    ```
*   **Alternative (Explicit Transformation Chain, for understanding/older DIALS):**
    ```python
    from scitbx import matrix
    import numpy as np

    def hkl_to_lab_q(expt, hkl_tuple_or_vec):
        if not isinstance(hkl_tuple_or_vec, matrix.col):
            hkl_vec = matrix.col(hkl_tuple_or_vec)
        else:
            hkl_vec = hkl_tuple_or_vec

        A_gonio_frame = matrix.sqr(expt.crystal.get_A())
        S = matrix.sqr(expt.goniometer.get_setting_rotation())
        F = matrix.sqr(expt.goniometer.get_fixed_rotation())
        # C for lab_x=gonio_x, lab_y=-gonio_z, lab_z=gonio_y (-90 deg rot around X)
        C_lab_from_gonio = matrix.sqr((1,0,0,  0,0,-1,  0,1,0)) 
        
        q_lab_scitbx = C_lab_from_gonio * S * F * A_gonio_frame * hkl_vec
        return np.array(q_lab_scitbx.elems)

    def get_q_bragg_from_reflection(refl, experiment):
        hkl_tuple = refl["miller_index"]
        q_bragg_np = hkl_to_lab_q(experiment, hkl_tuple)
        return q_bragg_np
    ```

**2. Importance of Using Refined Spot Coordinates:**
For calculating `q_pixel_recalculated`, it's crucial to use the refined spot positions, as these correspond to the refined geometric model.
In `consistency.py`:
```python
# x_obs_px, y_obs_px, _ = refl['xyzobs.px.value'] # Original
# USE REFINED COORDINATES:
try:
    x_cal_mm, y_cal_mm, _ = refl['xyzcal.mm']
    fast_px_cal, slow_px_cal = panel_model_current.millimeter_to_pixel((x_cal_mm, y_cal_mm))
    px_idx = int(round(fast_px_cal))
    py_idx = int(round(slow_px_cal))
except (KeyError, RuntimeError): # Fallback if 'xyzcal.mm' not present or conversion fails
    print(f"Warning: Using observed pixels for refl {i} due to missing/invalid xyzcal.mm.")
    x_obs_px, y_obs_px, _ = refl['xyzobs.px.value']
    px_idx = int(round(x_obs_px))
    py_idx = int(round(y_obs_px))
```

**3. Refinement Strategy in DIALS:**
A two-step process was found to be most robust for ensuring proper geometric refinement:
*   **Step 1: `dials.index` (Initial Indexing):**
    Run with default settings for stills (detector likely fixed or minimally refined). Output to `indexed_initial.expt`/`refl`.
    ```bash
    dials.index imported.expt strong.refl \
      indexing.known_symmetry.unit_cell="..." \
      indexing.known_symmetry.space_group="..." \
      output.experiments=indexed_initial.expt \
      output.reflections=indexed_initial.refl
    ```
*   **Step 2: `dials.refine` (Dedicated Geometric Refinement):**
    Use a PHIL file (`refine_detector_beam.phil`) or command-line overrides to explicitly refine detector geometry and beam parameters.
    `refine_detector_beam.phil`:
    ```phil
    refinement {
      parameterisation {
        detector { fix = None; hierarchy_level = 0 }
        beam { fix = None }
        crystal { fix = cell } # Initially fix cell to focus on geometry
        scan_varying = False
      }
      reflections { outlier { algorithm = null } }
    }
    ```
    Command:
    ```bash
    dials.refine indexed_initial.expt indexed_initial.refl \
      ../refine_detector_beam.phil \
      output.experiments=indexed_refined_detector.expt \
      output.reflections=indexed_refined_detector.refl
    ```
    Subsequently, one might perform a final refinement round allowing the crystal cell to refine as well (`crystal.fix = None`).

### 7. Results After Fixes

After implementing the correct `q_bragg` transformation (using `hkl_to_reciprocal_space_vec` or the explicit `C*S*F*A` chain) and using the output from a comprehensive `dials.refine` run (with detector and beam refinement):

*   **`|Δq|` Residuals:** The mean and median `|q_bragg - q_pixel|` values are expected to decrease dramatically, ideally to the order of `10⁻³` to `10⁻² Å⁻¹`.
*   **Heatmap:** The systematic top-to-bottom (or any other strong) gradient in the `|Δq|` heatmap should be eliminated. The plot should show predominantly "cool" colors (low `|Δq|`) with a more random distribution of any small remaining errors.
*   **Consistency:** All three q-vector sources should align:
    1.  `q_bragg_lab` (from `hkl_to_reciprocal_space_vec` or `C*S*F*A@hkl`)
    2.  `q_pixel` (from refined geometry and `xyzcal.mm`)
    3.  `q_from_table = refl['s1'] - beam.get_s0()` (using refined `s1` and beam)

### 8. Key Learnings & Best Practices

*   **Reference Frames are Paramount:** Always be explicit about and verify the coordinate reference frames of vectors and matrices. `crystal.get_A()` in DIALS returns A in the goniometer frame. Detector/beam models are in the lab frame.
*   **Static Goniometer Rotations Matter for Stills:** The full transformation chain `C*S*F` is necessary to map from the goniometer frame to the lab frame, even if `S` and `F` are identity for a simple still setup. The constant rotation `C` can be non-identity and crucial.
*   **Ad-hoc Fixes are Indicators:** Empirical fixes like sign flips (e.g., the Y-flip) are strong indicators of a deeper, unaddressed coordinate system or model mismatch.
*   **Leverage DIALS Convenience Functions:** Functions like `experiment.crystal.hkl_to_reciprocal_space_vec()` encapsulate complex transformations and are generally safer and more future-proof than manual calculations if their precise behavior is understood.
*   **Iterative Refinement and Diagnostics:** Complex geometric problems often require iterative refinement strategies and careful diagnostic checks (like `consistency.py` and heatmaps) at each stage.
*   **Comprehensive Model Refinement:** For highest accuracy, ensure detector geometry, beam parameters, and crystal parameters (orientation and cell) are all refined, ideally in a way that allows them to co-vary and settle into a self-consistent state. Using a dedicated `dials.refine` step often provides more control for this than `dials.index`'s internal refinement alone.
*   **Use Refined Spot Positions:** When comparing model predictions to observed data after refinement, use the refined spot coordinates (`xyzcal.mm` or `xyzcal.px.value`) rather than the initial observed ones (`xyzobs.px.value`).

### 9. Developer Guide: Ensuring q-vector Consistency

**Calculating `q_bragg` in the Lab Frame:**
The most reliable method is:
```python
q_vec_scitbx = experiment.crystal.hkl_to_reciprocal_space_vec(hkl_tuple)
q_bragg_np = np.array(q_vec_scitbx.elems)
```
If implementing manually, ensure the full `C*S*F*A_gonio` chain is used, where `C` is the constant goniometer-to-lab rotation for the specific setup.

**Calculating `q_pixel` from Geometry:**
(As in `consistency.py`'s `calculate_q_for_single_pixel` function)
1.  Get `k_in = np.array(beam_model.get_s0())`.
2.  Get `k_magnitude = 1.0 / beam_model.get_wavelength()`.
3.  For a given pixel (ideally from refined `xyzcal.mm` converted to pixel indices `px_idx, py_idx`):
    *   `P_lab = np.array(panel_model.get_pixel_lab_coord((px_idx, py_idx)))`.
    *   `D_scattered = P_lab - sample_origin` (sample origin usually `[0,0,0]`).
    *   `s1_lab = D_scattered / np.linalg.norm(D_scattered)`.
    *   `k_out = s1_lab * k_magnitude`.
    *   `q_pixel = k_out - k_in`.

**Recommended DIALS Refinement Strategies for Stills:**
1.  **Initial Indexing (`dials.index`):** Can use defaults or minimal refinement settings.
    ```bash
    dials.index imported.expt strong.refl \
      # ... other params ... \
      output.experiments=indexed_initial.expt output.reflections=indexed_initial.refl
    ```
2.  **Dedicated Refinement (`dials.refine`):** Crucial for geometric accuracy.
    `refine_all_geom.phil`:
    ```phil
    refinement {
      parameterisation {
        detector { fix = None; hierarchy_level = 0 }
        beam { fix = None }
        crystal { fix = None } # Allow cell and orientation to refine
        scan_varying = False
      }
      reflections { outlier { algorithm = auto } } # Or null if handled earlier
    }
    ```
    Command:
    ```bash
    dials.refine indexed_initial.expt indexed_initial.refl \
      ../refine_all_geom.phil \
      output.experiments=final_refined.expt output.reflections=final_refined.refl
    ```

**Verification Steps:**
*   Run `consistency.py` (using `hkl_to_reciprocal_space_vec` and `xyzcal.mm`).
*   Check `|Δq|` statistics (should be low).
*   Check heatmap (should be random, low error).
*   Check `Mean |q_pred_dials - q_pixel_recalc|` (should be very low if `xyzcal.mm` is used for `q_pixel`).

### 10. Project User Reference: Interpreting `consistency.py` Outputs

*   **`|Δq|` Statistics (Mean, Median, Max):**
    *   **Mean/Median < 0.01 - 0.02 Å⁻¹:** Excellent consistency.
    *   **Mean/Median 0.02 - 0.05 Å⁻¹:** Good, but some minor issues might exist.
    *   **Mean/Median > 0.05 Å⁻¹:** Indicates significant inconsistencies; review refinement and frame transformations.
*   **Heatmap:**
    *   **Uniformly "cool" colors (dark blue/purple):** Ideal.
    *   **Systematic gradients (e.g., top-to-bottom, center-to-edge):** Indicates uncorrected geometric mismodeling or frame issues. The nature of the gradient hints at the type of error (e.g., tilt, distance).
*   **Component Plots (if generated):**
    *   Trends in `Δqx`, `Δqy`, `Δqz` vs. pixel coordinates can further pinpoint the directional nature of the mismodeling. Flat lines around zero are ideal.
*   **`Mean |q_pred_dials - q_pixel_recalc|`:**
    *   Should be very small (`< 0.001 Å⁻¹` or better if `xyzcal.mm` is used correctly for `q_pixel`). If large, there's an issue in how `q_pixel` is being calculated relative to DIALS's own `s1` predictions or how refined positions are used.

### 11. Future Work & Recommendations

*   **DIALS Versioning:** Standardize on a recent, stable DIALS version (e.g., ≥ 3.12) where `hkl_to_reciprocal_space_vec` is reliable and well-tested for stills.
*   **Automated Testing:** Incorporate `consistency.py` (or a simplified version) into an automated testing pipeline for DIALS processing to catch regressions or issues with new datasets/setups. Set a threshold for acceptable mean `|Δq|`.
*   **Further Exploration of Subtle Residuals:** If minor systematic residuals persist even after these fixes, they might point to more complex detector effects (e.g., panel bowing, non-uniform pixel response) not captured by current models, or very subtle beam path issues.

### 12. Appendix: Example Script Snippets (Final Corrected Versions)

**`get_q_bragg_from_reflection` in `consistency.py` (Recommended):**
```python
from scitbx import matrix
import numpy as np

def get_q_bragg_from_reflection(refl, experiment):
    hkl_tuple = refl["miller_index"]
    q_vec_scitbx = experiment.crystal.hkl_to_reciprocal_space_vec(hkl_tuple)
    q_bragg_np = np.array(q_vec_scitbx.elems)
    return q_bragg_np
```

**Relevant part of `consistency.py` for `q_pixel` (using `xyzcal.mm`):**
```python
# ... inside loop, after getting panel_model_current ...
try:
    x_cal_mm, y_cal_mm, _ = refl['xyzcal.mm']
    fast_px_cal, slow_px_cal = panel_model_current.millimeter_to_pixel((x_cal_mm, y_cal_mm))
    px_idx = int(round(fast_px_cal))
    py_idx = int(round(slow_px_cal))
except (KeyError, RuntimeError):
    print(f"Warning: Using observed pixels for refl {i} due to missing/invalid xyzcal.mm.")
    x_obs_px, y_obs_px, _ = refl['xyzobs.px.value'] # Fallback
    px_idx = int(round(x_obs_px))
    py_idx = int(round(y_obs_px))

q_pixel_recalculated = calculate_q_for_single_pixel(
    current_experiment.beam, panel_model_current, px_idx, py_idx
)
```

**Example `refine_all_geom.phil` for `dials.refine`:**
```phil
refinement {
  parameterisation {
    detector {
      fix = None
      hierarchy_level = 0 // For single panel, refine its origin and orientation
    }
    beam {
      fix = None // Refine beam direction and wavelength
    }
    crystal {
      fix = None // Refine crystal orientation AND unit cell
    }
    scan_varying = False // Essential for stills
  }
  reflections {
    outlier {
      // algorithm = auto // 'auto' is often good for robust outlier rejection
      algorithm = null // If you prefer to handle outliers separately or trust input
    }
  }
  // target { // Optional: for more aggressive refinement if RMSDs are high
  //   rmsd_cutoff {
  //     value = 0.2 // Example: target RMSD in pixels
  //   }
  // }
}
```

