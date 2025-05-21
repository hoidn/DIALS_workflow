#!/usr/bin/env python
# extract_dials_data_for_eryx.py
import argparse
import os
import sys
import numpy as np
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family import flex
from scitbx import matrix
from iotbx.pdb import input as pdb_input 
from cctbx import uctbx, sgtbx 
import matplotlib.pyplot as plt 
import pickle 
import logging

from dxtbx.imageset import ImageSetFactory
from dials.model.data import Shoebox 
from dials.algorithms.shoebox import MaskCode

from scipy.constants import pi
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("extract_dials")

# --- Helper function from consistency.py or similar, for q_bragg calculation ---
def hkl_to_lab_q(experiment, hkl_vec):
    A = matrix.sqr(experiment.crystal.get_A())
    S = matrix.sqr(experiment.goniometer.get_setting_rotation())
    F = matrix.sqr(experiment.goniometer.get_fixed_rotation())
    C = matrix.sqr((1,0,0, 0,0,-1, 0,1,0))
    R_lab = C * S * F
    return R_lab * A * hkl_vec

def get_q_bragg_from_refl_data(miller_index_tuple, experiment_crystal):
    try:
        q_vec_scitbx = experiment_crystal.hkl_to_reciprocal_space_vec(miller_index_tuple)
        return np.array(q_vec_scitbx.elems)
    except AttributeError:
        hkl_vec = matrix.col(miller_index_tuple) 
        # experiment_crystal is a Crystal object, hkl_to_lab_q expects an Experiment object
        # This requires a slight refactor or passing the full experiment if this path is taken.
        # Assuming experiment_crystal has a reference to its parent experiment if needed by hkl_to_lab_q
        # For now, this specific path might not be hit if hkl_to_reciprocal_space_vec is standard.
        # Safest is to ensure DIALS version provides hkl_to_reciprocal_space_vec.
        # As a fallback, one might need to construct a temporary minimal experiment for hkl_to_lab_q.
        # This part of the original script was: get_q_bragg_from_reflection(refl, current_experiment)
        # which passed the full experiment. For consistency, let's assume we have the full experiment context here.
        # This helper is now more specific to needing only crystal model and HKL.
        # For this utility mode, it is safer to rely on hkl_to_reciprocal_space_vec from the crystal model.
        print("Warning: Falling back to manual q_bragg calculation. Ensure experiment context for hkl_to_lab_q is correct if this happens.")
        # Simplified: assuming experiment_crystal.get_experiment() exists or is the experiment itself
        # This line is problematic as Crystal object doesn't typically store the full experiment.
        # Let's assume the caller (main Bragg loop) passes the full experiment for this fallback.
        # This is redefined below as a nested function in the bragg processing part for clarity.
        raise NotImplementedError("Fallback for get_q_bragg_from_refl_data without direct hkl_to_reciprocal_space_vec needs Experiment object.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract data from DIALS processing for eryx. Supports pixel-centric diffuse data extraction and Bragg-centric utility mode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--experiment_file", required=True, help="Path to the DIALS experiment (.expt) file.")
    parser.add_argument("--reflection_file", help="Path to the DIALS reflection (.refl) file (required for data-source=bragg).")
    parser.add_argument("--image_files", nargs='+', help="Path(s) to CBF/image files (required for data-source=pixels).")
    parser.add_argument("--bragg_mask_file", help="Path to the Bragg mask (.pickle) file (mandatory if data-source=pixels).")
    parser.add_argument("--external_pdb_file", help="Path to an external PDB file for consistency check and reference.")
    parser.add_argument("--output_npz_file", required=True, help="Path for the output .npz file.")
    parser.add_argument("--data-source", choices=["pixels", "bragg"], default="pixels",
                        help="Source of data: 'pixels' for diffuse scattering from image pixels, or 'bragg' for utility extraction of Bragg peak data.")
    parser.add_argument("--min_res", type=float, help="Minimum resolution in Angstroms (d_max). Filters q_pixel or q_bragg.")
    parser.add_argument("--max_res", type=float, help="Maximum resolution in Angstroms (d_min). Filters q_pixel or q_bragg.")
    parser.add_argument("--min_intensity", type=float, help="Minimum intensity threshold for pixel data.")
    parser.add_argument("--max_intensity", type=float, help="Maximum intensity threshold for pixel data.")
    parser.add_argument("--min_isigi_bragg", type=float, default=0.0, help="Minimum I/sigma(I) threshold for data-source=bragg.")
    parser.add_argument("--intensity_column_bragg", default="intensity.sum.value", help="Reflection table column for intensity (for data-source=bragg).")
    parser.add_argument("--variance_column_bragg", default="intensity.sum.variance", help="Reflection table column for variance (for data-source=bragg).")
    parser.add_argument("--cell_length_tol", type=float, default=0.1, help="Tolerance for unit cell length comparison (Angstroms).")
    parser.add_argument("--cell_angle_tol", type=float, default=1.0, help="Tolerance for unit cell angle comparison (degrees).")
    parser.add_argument("--orient_tolerance_deg", type=float, default=1.0, help="Tolerance for crystal orientation comparison vs external PDB (degrees).")
    parser.add_argument("--gain", type=float, default=1.0, help="Detector gain value for variance estimation (if not in detector model).")
    parser.add_argument("--pixel_step", type=int, default=1, help="Step size for pixel sampling (1=all pixels). For data-source=pixels.")
    parser.add_argument("--lp_correction_enabled", action='store_true', help="Enable simplified LP correction (polarization factor based on 2-theta). default: disabled")
    # T3.1 Background subtraction arguments
    parser.add_argument("--subtract_background_value", type=float, default=None, help="Constant value to subtract from pixel intensities (for data-source=pixels).")
    # Future: --bg_subtract_method for more advanced methods

    parser.add_argument("--plot", action="store_true", help="Generate diagnostic plots.")
    parser.add_argument("--verbose", action='store_true', help="Print detailed information during processing.")
    
    args = parser.parse_args()
    if args.data_source == "pixels" and not args.image_files: parser.error("--image_files required for data-source=pixels")
    if args.data_source == "pixels" and not args.bragg_mask_file: parser.error("--bragg_mask_file required for data-source=pixels")
    if args.data_source == "bragg" and not args.reflection_file: parser.error("--reflection_file required for data-source=bragg")
    return args

def load_external_pdb(pdb_file_path):
    if not pdb_file_path or not os.path.exists(pdb_file_path): print(f"Warning: External PDB {pdb_file_path} not found."); return None
    try: pdb_inp = pdb_input(file_name=pdb_file_path); cs = pdb_inp.crystal_symmetry(); return cs if cs else print(f"Warning: No crystal symm in PDB {pdb_file_path}.")
    except Exception as e: print(f"Error loading PDB {pdb_file_path}: {e}"); return None

def calculate_misorientation(A1_matrix, A2_matrix):
    """
    Return the smallest rotation angle (deg) that aligns A1 to A2,
    *after* also testing the centrosymmetric inversion (−I).
    """
    def angle(a, b):
        A1 = matrix.sqr(a)
        A2 = matrix.sqr(b)
        try:
            A1_inv = A1.inverse()
        except RuntimeError: # Singular matrix
            return 180.0 # Max misorientation if inverse fails
        R = A2 * A1_inv
        trace_R = R.trace()
        # Ensure trace_R is within valid range for acos [-1, 1]
        # (trace(R) - 1) / 2 should be cos(angle)
        cos_angle = (trace_R - 1.0) / 2.0
        cos_angle_clipped = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle_clipped)
        return np.degrees(angle_rad)

    a = angle(A1_matrix, A2_matrix)
    a_inv = angle(A1_matrix, [-x for x in A2_matrix])  # inverted hand
    return min(a, a_inv)

def check_pdb_consistency(experiments, ext_pdb_symm, len_tol, ang_tol, orient_tol_deg, verbose_flag):
    if not ext_pdb_symm:
        logger.warning("External PDB symmetry not loaded, skipping consistency check.")
        return True # Or False, depending on how strict you want to be if PDB is missing

    overall_consistency_ok = True
    ext_uc = ext_pdb_symm.unit_cell()
    ext_sg_info = ext_pdb_symm.space_group_info()
    
    # Calculate reference A matrix from external PDB's cell (conventional setting U=I)
    try:
        B_pdb_conventional = matrix.sqr(ext_pdb_symm.unit_cell().fractionalization_matrix()).inverse()
        A_pdb_ref = B_pdb_conventional # This is B matrix for PDB (assuming U is identity)
    except Exception as e:
        logger.error(f"Error calculating PDB reference matrix: {e}")
        return False

    for i, exp in enumerate(experiments):
        logger.info(f"--- PDB Consistency Check for Experiment {i} ---")
        exp_crystal = exp.crystal
        exp_cs = exp_crystal.get_crystal_symmetry()
        exp_uc = exp_cs.unit_cell()

        if verbose_flag:
            logger.info(f"  DIALS Experiment Cell: {exp_uc.parameters()}")
            logger.info(f"  External PDB Cell: {ext_uc.parameters()}")
        
        # Cell similarity
        if not exp_uc.is_similar_to(
            other=ext_uc, 
            relative_length_tolerance=len_tol, 
            absolute_angle_tolerance=ang_tol
        ):
            logger.warning(f"  Exp {i} CELL MISMATCH (Tol: rel_len={len_tol}, abs_ang={ang_tol} deg)!")
            logger.warning(f"    DIALS Cell: {exp_uc.parameters()}")
            logger.warning(f"    PDB Cell: {ext_uc.parameters()}")
            overall_consistency_ok = False
        
        # Space group
        exp_sg_info = exp_cs.space_group_info()
        if exp_sg_info.type().number() != ext_sg_info.type().number():
            logger.warning(f"  Exp {i} SPACE GROUP MISMATCH!")
            logger.warning(f"    DIALS SG: {exp_sg_info.symbol_and_number()}")
            logger.warning(f"    PDB SG: {ext_sg_info.symbol_and_number()}")
            overall_consistency_ok = False

        # Orientation
        try:
            A_elements_tuple = exp_crystal.get_A() # This is UB from DIALS
            A_expt = matrix.sqr(A_elements_tuple) # scitbx.matrix.sqr object

            if verbose_flag:
                logger.info(f"  DIALS A matrix (UB_dials) elements: {list(A_expt)}")
                # Use str() instead of as_string() for safer printing
                logger.info(f"  DIALS A matrix (UB_dials): {str(A_expt)}")
                logger.info(f"  PDB B matrix (B_pdb_ref): {str(A_pdb_ref)}")

            # Calculate misorientation
            misorientation_deg = calculate_misorientation(A_expt, A_pdb_ref)
            
            if verbose_flag:
                logger.info(f"  Misorientation (DIALS vs PDB): {misorientation_deg:.3f} degrees")

            if misorientation_deg > orient_tol_deg:
                logger.warning(f"  Exp {i} ORIENTATION MISMATCH! Misorientation: {misorientation_deg:.3f} deg > tolerance {orient_tol_deg:.3f} deg.")
                overall_consistency_ok = False
        except Exception as e_orient:
            logger.error(f"  Error during orientation check for Exp {i}: {e_orient}")
            overall_consistency_ok = False
            
    if overall_consistency_ok and verbose_flag:
        logger.info("--- PDB cell, space group, and orientation consistency OK. ---")
    elif not overall_consistency_ok:
        logger.warning("--- PDB consistency check found issues. ---")
    return overall_consistency_ok

def calculate_q_for_pixel_batch(beam, panel, fast_coords, slow_coords):
    s0=np.array(beam.get_s0()); k_mag=np.linalg.norm(s0); q_out=[]
    lab_coords = panel.get_lab_coord_multiple(flex.vec2_double(list(zip(fast_coords,slow_coords))))
    for P_lab_tup in lab_coords: P_lab=np.array(P_lab_tup); norm_P=np.linalg.norm(P_lab); s1_dir = P_lab/norm_P if norm_P >1e-9 else np.array([0,0,1]); q_out.append(s1_dir*k_mag - s0)
    return np.array(q_out)

def q_to_resolution(q):
    mag = np.linalg.norm(q) if isinstance(q,np.ndarray) else q; return (2*pi/mag) if mag>1e-9 else float('inf')

def apply_lp_correction(intensities, variances, q_vecs, s0_vec, verb):
    """
    Apply Lorentz-polarization correction to intensities and their variances.
    
    IMPORTANT: This implementation assumes an unpolarized incident beam.
    The correction formula used is (1 + cos²(2θ))/2, which is the standard LP factor
    for unpolarized radiation. For a typical synchrotron source (horizontally polarized),
    a different correction would be required.
    
    Args:
        intensities: Array of intensity values to correct
        variances: Array of variance values to correct
        q_vecs: Array of q-vectors for each intensity point
        s0_vec: Incident beam vector
        verb: Verbose flag for logging
        
    Returns:
        tuple: (corrected_intensities, corrected_variances)
    """
    if verb: print("Applying LP corr...")
    s0_np = np.array(s0_vec); k2 = np.dot(s0_np,s0_np)
    if k2<1e-9: print("Warning: k_val_sq near zero in LP. Skipping."); return intensities, variances
    
    # The formula (1.0 + np.clip(1.0-(np.dot(q,q)/(2.0*k2)),-1.0,1.0)**2)/2.0 is for unpolarized radiation
    # where the term np.clip(1.0-(np.dot(q,q)/(2.0*k2)),-1.0,1.0) is cos(2θ)
    # For polarized synchrotron radiation, a different formula would be needed that accounts for
    # the polarization fraction and direction
    pol_factors = [(1.0 + np.clip(1.0-(np.dot(q,q)/(2.0*k2)),-1.0,1.0)**2)/2.0 for q in q_vecs]
    pol_factors = np.maximum(pol_factors, 1e-6)
    corr_i = intensities/pol_factors; corr_v = variances/(pol_factors**2)
    if verb: print(f"  LP corrected {len(corr_i)} points.")
    return corr_i, corr_v

def main():
    args = parse_args()
    print(f"Loading DIALS experiment file: {args.experiment_file}")
    if not os.path.exists(args.experiment_file): print(f"Error: Exp file {args.experiment_file} not found."); sys.exit(1)
    experiments = ExperimentListFactory.from_json_file(args.experiment_file, check_format=False)
    if not experiments: print("Error: No experiments in file."); sys.exit(1)
    
    experiment = experiments[0] # Default experiment for geometry

    ext_pdb_symm = load_external_pdb(args.external_pdb_file) if args.external_pdb_file else None
    if ext_pdb_symm: 
        print("Performing PDB consistency check...")
        if not check_pdb_consistency(experiments, ext_pdb_symm, args.cell_length_tol, args.cell_angle_tol, args.orient_tolerance_deg, args.verbose):
            print("PDB consistency check FAILED. See warnings.")

    all_q_data, all_i_data, all_var_data = [], [], []

    if args.data_source == "pixels":
        print("Pixel-centric mode")
        try:
            with open(args.bragg_mask_file, 'rb') as f: bragg_mask_tuple = pickle.load(f)
            try:  # DIALS ≤ 3.1
                imagesets = ImageSetFactory.from_filenames(args.image_files)
            except AttributeError:  # DIALS ≥ 3.2
                imagesets = ImageSetFactory.new(args.image_files)
        except Exception as e: print(f"Error loading pixel data inputs: {e}"); sys.exit(1)
        if not imagesets: print("Error: No image sets loaded."); sys.exit(1)

        imageset = imagesets[0]; detector = experiment.detector; beam = experiment.beam
        total_pix = sum((p.get_image_size()[0]//args.pixel_step)*(p.get_image_size()[1]//args.pixel_step) for p in detector)
        
        with tqdm(total=total_pix, desc="Pixels", unit="pix", disable=args.verbose) as pbar: # Corrected tqdm disable logic
            for frame_idx in range(len(imageset)):
                raw_data_tup = imageset.get_raw_data(frame_idx)
                for panel_idx, panel in enumerate(detector):
                    panel_data_flex = raw_data_tup[panel_idx]          # flex array straight from DIALS
                    panel_data_np = panel_data_flex.as_numpy_array()   # NumPy copy for later arithmetic
                    panel_bragg_mask_loaded = bragg_mask_tuple[panel_idx].as_numpy_array()
                    panel_trusted_mask = panel.get_trusted_range_mask(panel_data_flex).as_numpy_array()
                    
                    # Get dimensions from detector model
                    # panel.get_image_size() returns (fast, slow) but NumPy arrays are (slow, fast)
                    fs_detector, ss_detector = panel.get_image_size()  # Fast, Slow from detector model
                    expected_np_shape = (ss_detector, fs_detector)  # Expected NumPy shape (slow, fast)
                    
                    # Check if Bragg mask needs transposing (common issue with flex->numpy conversion)
                    panel_bragg_mask = panel_bragg_mask_loaded
                    if panel_bragg_mask_loaded.shape == (fs_detector, ss_detector):  # If it's (fast, slow)
                        logger.info(f"Panel {panel_idx}: Bragg mask dimensions appear transposed {panel_bragg_mask_loaded.shape}. Transposing to {expected_np_shape}.")
                        panel_bragg_mask = panel_bragg_mask_loaded.transpose()
                    
                    # Get dimensions of all arrays after potential transposing
                    ss_data, fs_data = panel_data_np.shape  # Dimensions from actual data array
                    ss_bragg_mask, fs_bragg_mask = panel_bragg_mask.shape  # After potential transpose
                    ss_trusted_mask, fs_trusted_mask = panel_trusted_mask.shape
                    
                    if args.verbose:
                        logger.info(f"Panel {panel_idx} DIMS (NumPy convention: slow,fast): "
                                    f"DetectorModel(s,f)=({ss_detector},{fs_detector}), "
                                    f"ImageData(s,f)=({ss_data},{fs_data}), "
                                    f"BraggMask(s,f)=({ss_bragg_mask},{fs_bragg_mask}), "
                                    f"TrustedMask(s,f)=({ss_trusted_mask},{fs_trusted_mask})")
                    
                    # Check for dimension mismatches and handle appropriately
                    if panel_data_np.shape != expected_np_shape or \
                       panel_bragg_mask.shape != expected_np_shape or \
                       panel_trusted_mask.shape != expected_np_shape:
                        logger.error(f"CRITICAL DIMENSION MISMATCH for Panel {panel_idx} after adjustments!")
                        logger.error(f"  Expected NumPy shape (slow,fast): {expected_np_shape}")
                        logger.error(f"  Got: ImageData{panel_data_np.shape}, BraggMask{panel_bragg_mask.shape}, TrustedMask{panel_trusted_mask.shape}")
                        logger.error(f"  Skipping panel {panel_idx} due to dimension mismatch")
                        continue  # Skip this panel
                    
                    f_coords,s_coords,p_i,p_v = [],[],[],[]
                    
                    # Panel-level counters for diagnostics
                    panel_total_pixels = 0
                    panel_rejected_by_bragg_mask = 0
                    panel_rejected_by_trusted_range = 0
                    panel_rejected_by_min_intensity = 0
                    panel_rejected_by_max_intensity = 0
                    
                    if args.verbose:
                        logger.info(f"Panel {panel_idx}: Processing {ss}x{fs} pixels with step {args.pixel_step}")
                        logger.info(f"  Min intensity filter: {args.min_intensity}")
                        logger.info(f"  Max intensity filter: {args.max_intensity}")
                        logger.info(f"  Background subtraction: {args.subtract_background_value}")
                    
                    for sl_idx in range(0,ss_detector,args.pixel_step):
                        for ft_idx in range(0,fs_detector,args.pixel_step):
                            pbar.update(1)
                            panel_total_pixels += 1
                            
                            # Check Bragg mask - use try/except to catch any index errors
                            try:
                                if panel_bragg_mask[sl_idx,ft_idx]:
                                    panel_rejected_by_bragg_mask += 1
                                    if args.verbose and panel_rejected_by_bragg_mask < 10:  # Log first few rejections
                                        logger.info(f"  P({panel_idx},{sl_idx},{ft_idx}): REJECTED by Bragg mask")
                                    continue
                            except IndexError as e:
                                logger.error(f"IndexError accessing Bragg mask at ({sl_idx},{ft_idx}): {e}")
                                logger.error(f"  Panel {panel_idx} Bragg mask shape: {panel_bragg_mask.shape}")
                                logger.error(f"  This should not happen if dimensions were verified. Skipping pixel.")
                                continue
                            
                            # Check trusted range - use try/except to catch any index errors
                            try:
                                if not panel_trusted_mask[sl_idx,ft_idx]:
                                    panel_rejected_by_trusted_range += 1
                                    if args.verbose and panel_rejected_by_trusted_range < 10:  # Log first few rejections
                                        logger.info(f"  P({panel_idx},{sl_idx},{ft_idx}): REJECTED by detector trusted range")
                                        # Print the actual pixel value for debugging
                                        logger.info(f"    Pixel value: {panel_data_np[sl_idx,ft_idx]}, Trusted range: {panel.get_trusted_range()}")
                                    continue
                            except IndexError as e:
                                logger.error(f"IndexError accessing trusted mask at ({sl_idx},{ft_idx}): {e}")
                                logger.error(f"  Panel {panel_idx} trusted mask shape: {panel_trusted_mask.shape}")
                                logger.error(f"  This should not happen if dimensions were verified. Skipping pixel.")
                                continue
                            
                            # Check intensity filters
                            intensity = panel_data_np[sl_idx,ft_idx]
                            original_intensity = intensity
                            
                            if args.subtract_background_value is not None:
                                intensity -= args.subtract_background_value
                            
                            if args.min_intensity is not None and intensity < args.min_intensity:
                                panel_rejected_by_min_intensity += 1
                                if args.verbose and panel_rejected_by_min_intensity < 10:  # Log first few rejections
                                    logger.info(f"  P({panel_idx},{sl_idx},{ft_idx}): OrigI={original_intensity:.1f}, BGSubI={intensity:.1f}. REJECTED by min_intensity ({args.min_intensity})")
                                continue
                            
                            if args.max_intensity is not None and intensity > args.max_intensity:
                                panel_rejected_by_max_intensity += 1
                                if args.verbose and panel_rejected_by_max_intensity < 10:  # Log first few rejections
                                    logger.info(f"  P({panel_idx},{sl_idx},{ft_idx}): OrigI={original_intensity:.1f}, BGSubI={intensity:.1f}. REJECTED by max_intensity ({args.max_intensity})")
                                continue
                            
                            # Pixel passed all initial filters
                            f_coords.append(ft_idx); s_coords.append(sl_idx); p_i.append(intensity)
                            # Variance: if BG subtracted, var(I_obs - BG_const) = var(I_obs). If BG has variance, add var(BG).
                            # Assuming BG_const for now. If image is pre-subtracted, this variance is less accurate.
                            p_v.append(original_intensity / args.gain if args.gain > 0 else original_intensity)

                    # Log panel filter statistics
                    if args.verbose:
                        logger.info(f"Panel {panel_idx} filter statistics:")
                        logger.info(f"  Total pixels processed: {panel_total_pixels}")
                        logger.info(f"  Rejected by Bragg mask: {panel_rejected_by_bragg_mask} ({panel_rejected_by_bragg_mask/panel_total_pixels*100:.1f}%)")
                        logger.info(f"  Rejected by trusted range: {panel_rejected_by_trusted_range} ({panel_rejected_by_trusted_range/panel_total_pixels*100:.1f}%)")
                        logger.info(f"  Rejected by min intensity: {panel_rejected_by_min_intensity} ({panel_rejected_by_min_intensity/panel_total_pixels*100:.1f}%)")
                        logger.info(f"  Rejected by max intensity: {panel_rejected_by_max_intensity} ({panel_rejected_by_max_intensity/panel_total_pixels*100:.1f}%)")
                        logger.info(f"  Candidates after initial filters: {len(f_coords)}")
                    
                    if not f_coords: 
                        if args.verbose:
                            logger.info(f"Panel {panel_idx}: No candidate pixels after initial filters.")
                        continue
                        
                    q_batch = calculate_q_for_pixel_batch(beam, panel, f_coords, s_coords)
                    accepted_idx = np.full(len(q_batch), True)
                    
                    # Resolution filter statistics
                    num_before_res_filter = len(q_batch)
                    num_rejected_by_res_filter = 0
                    
                    if args.min_res is not None or args.max_res is not None:
                        d_spacings = np.array([q_to_resolution(q) for q in q_batch])
                        
                        if args.verbose:
                            logger.info(f"  Resolution range: d_max={args.min_res}, d_min={args.max_res}")
                            logger.info(f"  d-spacing range in data: {np.min(d_spacings):.2f}A - {np.max(d_spacings):.2f}A")
                        
                        if args.min_res: 
                            min_res_mask = (d_spacings <= args.min_res)
                            num_rejected_by_min_res = np.sum(~min_res_mask)
                            if args.verbose:
                                logger.info(f"  Rejected by min_res: {num_rejected_by_min_res} ({num_rejected_by_min_res/len(d_spacings)*100:.1f}%)")
                            accepted_idx &= min_res_mask
                            
                        if args.max_res: 
                            max_res_mask = (d_spacings >= args.max_res)
                            num_rejected_by_max_res = np.sum(~max_res_mask)
                            if args.verbose:
                                logger.info(f"  Rejected by max_res: {num_rejected_by_max_res} ({num_rejected_by_max_res/len(d_spacings)*100:.1f}%)")
                            accepted_idx &= max_res_mask
                        
                        num_rejected_by_res_filter = np.sum(~accepted_idx)
                    
                    num_after_res_filter = np.sum(accepted_idx)
                    if args.verbose:
                        logger.info(f"  Candidates before res filter: {num_before_res_filter}")
                        logger.info(f"  Rejected by res filter: {num_rejected_by_res_filter}")
                        logger.info(f"  Accepted after res filter: {num_after_res_filter}")
                    
                    all_q_data.extend(q_batch[accepted_idx]); all_i_data.extend(np.array(p_i)[accepted_idx]); all_var_data.extend(np.array(p_v)[accepted_idx])
                    
                    if args.verbose and num_after_res_filter > 0:
                        logger.info(f"  Added {num_after_res_filter} points from panel {panel_idx}. Total collected so far: {len(all_q_data)}")
        
        all_q_data=np.array(all_q_data); all_i_data=np.array(all_i_data); all_var_data=np.array(all_var_data)
        if args.lp_correction_enabled and len(all_q_data)>0: all_i_data,all_var_data = apply_lp_correction(all_i_data,all_var_data,all_q_data,beam.get_s0(),args.verbose)
        print(f"Collected {len(all_q_data)} data points from pixels.")
        
        # Print detector trusted range for debugging
        if args.verbose:
            for i, panel in enumerate(detector):
                trusted_range = panel.get_trusted_range()
                logger.info(f"Panel {i} trusted range: {trusted_range}")
                
        # Print summary of filter rejections
        if args.verbose and len(all_q_data) == 0:
            logger.info("\nDEBUGGING SUMMARY: No pixels passed all filters!")
            logger.info("Possible issues to check:")
            logger.info("1. Bragg mask might be too aggressive - check with dials.image_viewer")
            logger.info("2. Detector trusted range might be too restrictive")
            logger.info("3. Intensity filters might be inappropriate for your data")
            logger.info("4. Resolution filters might be too narrow")
            logger.info("Try running with --min_intensity=None --max_intensity=None to disable intensity filtering")
            
        if args.verbose and len(all_q_data) > 0: 
            print("First few (q,I,var):"); 
            [print(f"  {all_q_data[i].tolist()}, {all_i_data[i]:.2f}, {all_var_data[i]:.2f}") for i in range(min(5,len(all_q_data)))]
    
    elif args.data_source == "bragg":
        print("WARNING: Bragg data utility mode. Output NOT for eryx diffuse intensity fitting.")
        if not os.path.exists(args.reflection_file): print(f"Error: Reflection file {args.reflection_file} not found."); sys.exit(1)
        reflections = flex.reflection_table.from_file(args.reflection_file)
        if 'flags' in reflections: # Ensure flags column exists before trying to use it
            indexed_sel = reflections.get_flags(reflections.flags.indexed)
            reflections = reflections.select(indexed_sel)
        else:
            print("Warning: No 'flags' column in reflection table. Processing all reflections.")
        
        print(f"Processing {len(reflections)} Bragg reflections.")
        # Define q_bragg helper that uses the correct experiment context
        def get_q_bragg_for_this_experiment(miller_index_tuple, current_experiment_obj):
            try: return np.array(current_experiment_obj.crystal.hkl_to_reciprocal_space_vec(miller_index_tuple).elems)
            except AttributeError: hkl_vec=matrix.col(miller_index_tuple); return np.array(hkl_to_lab_q(current_experiment_obj, hkl_vec).elems)

        for i in tqdm(range(len(reflections)), desc="Bragg Refs", unit="refl", disable=not args.verbose):
            refl = reflections[i]
            try: exp_id = refl['id'] if 'id' in refl else 0 
            except Exception: exp_id = 0 # Default if id access fails
            
            current_experiment = experiments[exp_id if 0 <= exp_id < len(experiments) else 0]
            
            if 'miller_index' not in refl: continue
            q_bragg = get_q_bragg_for_this_experiment(refl['miller_index'], current_experiment)
            if q_bragg is None: continue
            
            d_spacing = q_to_resolution(q_bragg)
            if (args.min_res is not None and d_spacing > args.min_res) or \
               (args.max_res is not None and d_spacing < args.max_res): continue
            
            try:
                intensity = refl[args.intensity_column_bragg]; variance = refl[args.variance_column_bragg]
            except KeyError as e:
                if args.verbose: print(f"Skipping refl {i} due to missing column: {e}")
                continue

            if variance <= 0: continue 
            isigi = intensity / np.sqrt(variance)
            if isigi < args.min_isigi_bragg: continue
            
            all_q_data.append(q_bragg); all_i_data.append(intensity); all_var_data.append(variance)
        all_q_data=np.array(all_q_data); all_i_data=np.array(all_i_data); all_var_data=np.array(all_var_data)
        print(f"Collected {len(all_q_data)} (q, I, var) data points from Bragg reflections.")

    if len(all_q_data) > 0:
        print(f"\\nSaving data to {args.output_npz_file}")
        np.savez_compressed(args.output_npz_file, q_vectors=all_q_data, intensities=all_i_data, variances=all_var_data, gain=args.gain)
        print("Data saved successfully.")
    else: print("\\nNo data collected to save.")
    print("Script finished.")

if __name__ == "__main__":
    main()
