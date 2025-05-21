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

from dxtbx.imageset import ImageSetFactory
from dials.model.data import Shoebox 
from dials.algorithms.shoebox import MaskCode

from scipy.constants import pi
from tqdm import tqdm

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

def check_pdb_consistency(experiments, ext_pdb_symm, len_tol, ang_tol, orient_tol_deg, verb):
    if not ext_pdb_symm: return True
    ok = True
    # The PDB crystal symmetry object (ext_pdb_symm) gives cell & space group.
    # For orientation, we need its A matrix. cctbx.crystal.symmetry objects
    # don't store an explicit orientation unless they were created from a model
    # that has one (e.g. a dxtbx Crystal object).
    # We assume the external PDB provides cell parameters, and DIALS orients its model
    # relative to this PDB if used as reference_geometry.
    # The A matrix from ext_pdb_symm will be based on its unit cell and a conventional setting.
    
    # Get B matrix for external PDB (conventional setting)
    B_pdb_conventional = matrix.sqr(ext_pdb_symm.unit_cell().orthogonalization_matrix())
    # For a PDB used as reference_geometry, DIALS's refined U matrix should be close to identity
    # if the PDB itself was in a standard setting.
    # So A_pdb_ref = U_ref * B_pdb_conventional. If U_ref is identity, A_pdb_ref = B_pdb_conventional.
    # This is an approximation if the PDB had a non-standard orientation that DIALS tried to match.
    # A more robust way for PDBs *without* explicit orientation:
    # The key is that DIALS's refinement used the PDB as reference_geometry.
    # So the DIALS crystal model's A matrix IS the one to compare against a hypothetical A matrix
    # formed from the PDB's cell constants and an *ideal* orientation (e.g. U=identity).
    
    # Let's assume DIALS' A matrix should be compared against the PDB's conventional A matrix.
    # This means we expect the DIALS U matrix to be identity if the PDB was conventionally aligned.
    # For external PDB, A = U_pdb * B_pdb. If PDB has no specific U, U_pdb = Identity.
    # So, A_pdb_ref = B_pdb_conventional (assuming U_pdb = Identity)
    A_pdb_ref = B_pdb_conventional


    for i, exp in enumerate(experiments):
        exp_crystal = exp.crystal
        exp_cs = exp_crystal.get_crystal_symmetry()
        exp_uc = exp_cs.unit_cell()
        ext_uc = ext_pdb_symm.unit_cell()
        
        if verb: 
            print(f"PDB Check Exp {i}:\n  Exp Cell: {exp_uc}\n  PDB Cell: {ext_uc}")
            
        # Cell similarity
        if not exp_uc.is_similar_to(
            other=ext_uc, 
            relative_length_tolerance=len_tol,
            absolute_angle_tolerance=ang_tol
        ):
            print(f"Warning Exp {i} CELL MISMATCH! Exp: {exp_uc} vs PDB: {ext_uc}")
            ok = False
            
        # Space group
        exp_sg_info = exp_cs.space_group_info()
        ext_sg_info = ext_pdb_symm.space_group_info()
        if exp_sg_info.type().number() != ext_sg_info.type().number():
            print(f"Warning Exp {i} SPACE GROUP MISMATCH! Exp: {exp_sg_info.symbol_and_number()} vs PDB: {ext_sg_info.symbol_and_number()}")
            ok = False

        # Orientation
        # A_expt is the orientation matrix from DIALS experiment (refined)
        A_expt = matrix.sqr(exp_crystal.get_A()) 
        
        # If the PDB defines a crystal (e.g. from CRYST1 card), it has cell params.
        # If it also has coordinates, it implies an orientation.
        # However, ext_pdb_symm is a cctbx.crystal_symmetry, usually derived from CRYST1 only.
        # For orientation, we need to construct an A matrix for the PDB.
        # If the PDB was used as reference_geometry, DIALS crystal model (A_expt)
        # should be aligned to it. The PDB itself, if it has coordinates, defines its own A matrix.
        # A simple PDB file might not have an explicit U matrix.
        # For now, let's use the A matrix from the PDB's unit cell parameters in a standard setting (U=I).
        # A_pdb_standard_setting = matrix.sqr(ext_pdb_symm.unit_cell().orthogonalization_matrix())
        # This A_pdb_standard_setting is what A_pdb_ref is above.

        misorientation_deg = calculate_misorientation(A_expt, A_pdb_ref)
        
        if verb:
            print(f"  Exp A matrix: {A_expt.round(5).as_string()}")
            # print(f"  PDB Ref A matrix (from PDB cell, U=I): {A_pdb_ref.round(5).as_string()}")
            print(f"  Misorientation (Exp vs PDB standard cell): {misorientation_deg:.3f} degrees")

        if misorientation_deg > orient_tol_deg:
            print(f"Warning Exp {i} ORIENTATION MISMATCH! Misorientation: {misorientation_deg:.3f} deg > tolerance {orient_tol_deg:.3f} deg.")
            ok = False
            
    if ok and verb: print("PDB cell, space group, and orientation consistency OK.")
    elif not ok and verb: print("PDB consistency check found issues.")
    return ok

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
                    panel_data_np = raw_data_tup[panel_idx].as_numpy_array()
                    panel_bragg_mask = bragg_mask_tuple[panel_idx].as_numpy_array()
                    panel_trusted_mask = panel.get_trusted_range_mask(panel_data_np).as_numpy_array()
                    fs, ss = panel.get_image_size(); f_coords,s_coords,p_i,p_v = [],[],[],[]
                    for sl_idx in range(0,ss,args.pixel_step):
                        for ft_idx in range(0,fs,args.pixel_step):
                            pbar.update(1)
                            if panel_bragg_mask[sl_idx,ft_idx] or not panel_trusted_mask[sl_idx,ft_idx]: continue
                            
                            intensity = panel_data_np[sl_idx,ft_idx]
                            if args.subtract_background_value is not None:
                                intensity -= args.subtract_background_value
                            
                            if (args.min_intensity is not None and intensity < args.min_intensity) or \
                               (args.max_intensity is not None and intensity > args.max_intensity): continue
                            
                            f_coords.append(ft_idx); s_coords.append(sl_idx); p_i.append(intensity)
                            # Variance: if BG subtracted, var(I_obs - BG_const) = var(I_obs). If BG has variance, add var(BG).
                            # Assuming BG_const for now. If image is pre-subtracted, this variance is less accurate.
                            p_v.append(panel_data_np[sl_idx,ft_idx] / args.gain if args.gain > 0 else panel_data_np[sl_idx,ft_idx])

                    if not f_coords: continue
                    q_batch = calculate_q_for_pixel_batch(beam, panel, f_coords, s_coords)
                    accepted_idx = np.full(len(q_batch), True)
                    if args.min_res is not None or args.max_res is not None:
                        d_spacings = np.array([q_to_resolution(q) for q in q_batch])
                        if args.min_res: accepted_idx &= (d_spacings <= args.min_res)
                        if args.max_res: accepted_idx &= (d_spacings >= args.max_res)
                    
                    all_q_data.extend(q_batch[accepted_idx]); all_i_data.extend(np.array(p_i)[accepted_idx]); all_var_data.extend(np.array(p_v)[accepted_idx])
        
        all_q_data=np.array(all_q_data); all_i_data=np.array(all_i_data); all_var_data=np.array(all_var_data)
        if args.lp_correction_enabled and len(all_q_data)>0: all_i_data,all_var_data = apply_lp_correction(all_i_data,all_var_data,all_q_data,beam.get_s0(),args.verbose)
        print(f"Collected {len(all_q_data)} data points from pixels.")
        if args.verbose and len(all_q_data) > 0: print("First few (q,I,var):"); [print(f"  {all_q_data[i].tolist()}, {all_i_data[i]:.2f}, {all_var_data[i]:.2f}") for i in range(min(5,len(all_q_data)))]
    
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
