#!/usr/bin/env python

import os
import sys
import subprocess
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("dials_pipeline_py")

# --- Configuration for extract_dials_data_for_eryx.py defaults ---
# These can be overridden by adding more CLI args to this script and passing them on
EXTRACT_DEFAULTS = {
    "gain": "1.0",
    "cell_length_tol": "0.1",
    "cell_angle_tol": "1.0",
    "orient_tolerance_deg": "0.5",
    "min_res": None, # Example: "2.0"
    "max_res": None, # Example: "50.0"
    "min_intensity": None, # Example: "10"
    "max_intensity": None, # Example: "60000"
    "pixel_step": "1",
    "lp_correction_enabled": False,
    "subtract_background_value": None,
    "plot": False,
    "verbose_extract": False,
    "min_isigi_bragg": "0.0", # For data-source=bragg
    "intensity_column_bragg": "intensity.sum.value", # For data-source=bragg
    "variance_column_bragg": "intensity.sum.variance" # For data-source=bragg
}
# --------------------------------------------------------------------

def run_command(cmd, log_file=None, work_dir=None):
    """Run a command and optionally log output to a file."""
    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info(f"In working directory: {work_dir if work_dir else os.getcwd()}")

    stdout_capture = None
    
    if log_file:
        # Ensure log_file path is absolute or relative to the intended directory
        log_path = os.path.join(work_dir, log_file) if work_dir and not os.path.isabs(log_file) else log_file
        with open(log_path, 'w') as f:
            process = subprocess.run(cmd, text=True, stdout=f, stderr=subprocess.STDOUT, cwd=work_dir)
    else:
        process = subprocess.run(cmd, text=True, capture_output=True, cwd=work_dir)
        stdout_capture = process.stdout

    if process.returncode != 0:
        logger.error(f"Command failed with exit code {process.returncode}")
        if not log_file and process.stderr: # process.stdout will contain both if STDOUT redirected
            logger.error(f"Error output: {process.stdout}") # stdout contains merged output
        elif not log_file and process.stdout:
             logger.info(f"Output: {process.stdout}")
        return False, stdout_capture
    
    logger.info(f"Command successful: {' '.join(cmd[:2])}...")
    return True, stdout_capture

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DIALS processing pipeline for a single CBF image/wedge, preparing data for eryx.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("cbf_file", help="Path to the CBF file (or first file of a wedge if DIALS import handles templates).")
    parser.add_argument("--external_pdb", help="Path to external PDB file for reference.", required=True)
    parser.add_argument("--output_dir", help="Base output directory (default: derived from CBF filename in current dir).", default=None)
    
    # DIALS PHIL files
    parser.add_argument("--find_spots_phil", help="PHIL file for dials.find_spots.", default="./find_spots.phil")
    parser.add_argument("--indexing_phil_file", help="Additional PHIL file for dials.index.", default=None)
    parser.add_argument("--refinement_phil_file", help="Additional PHIL file for dials.refine.", default="./refine_detector.phil")
    parser.add_argument("--mask_generation_phil_file", help="Additional PHIL file for dials.generate_mask.", default=None)

    # DIALS parameters
    parser.add_argument("--min_spot_size", type=int, help="Minimum spot size for spot finding.", default=3)
    # Add other DIALS specific params if needed, e.g., for indexing, refinement strategies not covered by PHILs

    # Extraction script parameters (can add all from extract_dials_data_for_eryx.py here)
    parser.add_argument("--extract_gain", type=float, default=EXTRACT_DEFAULTS["gain"], help="Gain for extraction script.")
    parser.add_argument("--extract_orient_tol_deg", type=float, default=EXTRACT_DEFAULTS["orient_tolerance_deg"], help="Orientation tolerance for extraction script.")
    # ... add more extraction script args as needed ...

    parser.add_argument("--skip_post_processing", action="store_true", help="Skip post-DIALS Python script steps (extraction, diagnostics).")
    parser.add_argument("--run_diagnostics", action="store_true", help="Run diagnostic Python scripts (q-map, consistency check).")

    return parser.parse_args()

def main():
    args = parse_args()
    
    orig_dir = os.getcwd() # Store original directory

    # Validate input files relative to orig_dir
    abs_cbf_file = os.path.abspath(os.path.join(orig_dir, args.cbf_file))
    abs_external_pdb = os.path.abspath(os.path.join(orig_dir, args.external_pdb))
    abs_find_spots_phil = os.path.abspath(os.path.join(orig_dir, args.find_spots_phil))

    if not os.path.exists(abs_cbf_file):
        logger.error(f"CBF file not found: {abs_cbf_file}"); return 1
    if not os.path.exists(abs_external_pdb):
        logger.error(f"External PDB file not found: {abs_external_pdb}"); return 1
    if not os.path.exists(abs_find_spots_phil):
        logger.error(f"Find spots PHIL file not found: {abs_find_spots_phil}"); return 1
        
    # Parse PDB to extract unit cell and space group
    pdb_unit_cell_str = None
    pdb_space_group_str = None
    overall_success = True
    
    try:
        from iotbx.pdb import input as pdb_input_loader
        from cctbx import uctbx

        logger.info(f"Parsing external PDB for cell/SG: {abs_external_pdb}")
        pdb_io = pdb_input_loader(file_name=abs_external_pdb)
        crystal_symmetry_from_pdb = pdb_io.crystal_symmetry()
        if crystal_symmetry_from_pdb and crystal_symmetry_from_pdb.unit_cell() and crystal_symmetry_from_pdb.space_group_info():
            pdb_unit_cell_str = ",".join(map(str, crystal_symmetry_from_pdb.unit_cell().parameters()))
            pdb_space_group_str = crystal_symmetry_from_pdb.space_group_info().type().lookup_symbol().replace(" ","")
            logger.info(f"  Extracted Unit Cell: {pdb_unit_cell_str}")
            logger.info(f"  Extracted Space Group: {pdb_space_group_str}")
        else:
            logger.error(f"Could not extract valid crystal symmetry (cell/SG) from PDB: {abs_external_pdb}")
            overall_success = False
    except Exception as e:
        logger.error(f"Error parsing external PDB {abs_external_pdb} for cell/SG: {e}")
        overall_success = False
        
    # Check success before proceeding
    if not overall_success:
        logger.error("Exiting due to PDB parsing failure.")
        return 1

    # Setup output directory (relative to orig_dir if args.output_dir is relative)
    base_name = os.path.basename(args.cbf_file).rsplit('.', 1)[0]
    if args.output_dir:
        # If output_dir is absolute, use it. If relative, it's relative to orig_dir.
        work_dir_base = os.path.join(orig_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    else:
        work_dir_base = os.path.join(orig_dir, f"{base_name}_processed_py") # Default output dir

    # Specific work directory for this CBF file's processing outputs
    # This script processes one CBF (or one DIALS import unit) at a time.
    # If work_dir_base is meant to hold multiple such processing dirs, adjust logic.
    # For now, assume work_dir_base is THE directory for this one CBF.
    abs_work_dir = os.path.abspath(work_dir_base) 
    
    logger.info(f"Processing {abs_cbf_file}")
    logger.info(f"Using external PDB: {abs_external_pdb}")
    logger.info(f"Output directory: {abs_work_dir}")
    
    os.makedirs(abs_work_dir, exist_ok=True)
    # DIALS commands will run with abs_work_dir as their cwd
    
    overall_success = True
    
    try:
        # --- DIALS Processing Steps ---
        # Step 1: dials.import
        logger.info("--- Step 1: dials.import ---")
        import_cmd = ["dials.import", abs_cbf_file, "output.experiments=imported.expt"]
        overall_success, _ = run_command(import_cmd, "dials.import.log", work_dir=abs_work_dir)
        if not overall_success or not os.path.exists(os.path.join(abs_work_dir, "imported.expt")):
            logger.error("dials.import failed or imported.expt not created."); raise RuntimeError("DIALS Import failed")

        # Step 2: dials.find_spots
        logger.info("--- Step 2: dials.find_spots ---")
        find_spots_cmd = [
            "dials.find_spots", "imported.expt", abs_find_spots_phil,
            f"spotfinder.filter.min_spot_size={args.min_spot_size}",
            "output.reflections=strong.refl", "output.shoeboxes=True" # Shoeboxes useful for diagnostics
        ]
        overall_success, _ = run_command(find_spots_cmd, "dials.find_spots.log", work_dir=abs_work_dir)
        if not overall_success or not os.path.exists(os.path.join(abs_work_dir, "strong.refl")):
            logger.error("dials.find_spots failed or strong.refl not created."); raise RuntimeError("DIALS Find Spots failed")

        # Step 3: dials.index (Direct call)
        logger.info("--- Step 3: dials.index ---")
        
        # Define output names for this step, matching the "old working" script
        current_indexed_expt_name = "indexed_initial.expt"
        current_indexed_refl_name = "indexed_initial.refl"
        
        index_cmd = [
            "dials.index", "imported.expt", "strong.refl",
            f"indexing.known_symmetry.unit_cell={pdb_unit_cell_str}",
            f"indexing.known_symmetry.space_group={pdb_space_group_str}",
            # Add other common/useful indexing parameters for stills
            "indexing.stills.indexer=stills", # Ensure stills indexer is used
            f"output.experiments={current_indexed_expt_name}",
            f"output.reflections={current_indexed_refl_name}"
        ]
        
        if args.indexing_phil_file and os.path.exists(os.path.join(orig_dir, args.indexing_phil_file)):
            index_cmd.append(os.path.abspath(os.path.join(orig_dir, args.indexing_phil_file)))
        
        overall_success, _ = run_command(index_cmd, "dials.index.log", work_dir=abs_work_dir)
        if not overall_success or not os.path.exists(os.path.join(abs_work_dir, current_indexed_expt_name)):
            logger.error(f"dials.index failed or {current_indexed_expt_name} not created."); raise RuntimeError("DIALS Index failed")
            
        # Step 4: dials.refine
        logger.info("--- Step 4: dials.refine ---")
        
        # Define output names for this step, matching the "old working" script
        final_refined_expt_name = "indexed_refined_detector.expt"
        final_refined_refl_name = "indexed_refined_detector.refl"
        
        refine_cmd = [
            "dials.refine", current_indexed_expt_name, current_indexed_refl_name,
            f"output.experiments={final_refined_expt_name}", 
            f"output.reflections={final_refined_refl_name}"
        ]
        
        # Add the refinement PHIL file
        if args.refinement_phil_file and os.path.exists(os.path.join(orig_dir, args.refinement_phil_file)):
            abs_refinement_phil = os.path.abspath(os.path.join(orig_dir, args.refinement_phil_file))
            refine_cmd.append(abs_refinement_phil)
            logger.info(f"Using refinement PHIL: {abs_refinement_phil}")
        else:
            logger.warning(f"Refinement PHIL file not specified or not found. Adding 'refinement.parameterisation.crystal.fix=cell' manually.")
            refine_cmd.append("refinement.parameterisation.crystal.fix=cell")

        overall_success, _ = run_command(refine_cmd, "dials.refine.log", work_dir=abs_work_dir)
        if not overall_success or not os.path.exists(os.path.join(abs_work_dir, final_refined_expt_name)):
            logger.error(f"dials.refine failed or {final_refined_expt_name} not created."); raise RuntimeError("DIALS Refine failed")
            
        # Step 5: dials.generate_mask
        logger.info("--- Step 5: dials.generate_mask ---")
        mask_cmd = [
            "dials.generate_mask", final_refined_expt_name, f"reflections={final_refined_refl_name}",
            "output.mask=bragg_mask.pickle"
            # Add parameters for mask generation if needed, e.g., border, d_min
        ]
        if args.mask_generation_phil_file and os.path.exists(os.path.join(orig_dir, args.mask_generation_phil_file)):
            mask_cmd.append(os.path.abspath(os.path.join(orig_dir, args.mask_generation_phil_file)))

        overall_success, _ = run_command(mask_cmd, "dials.generate_mask.log", work_dir=abs_work_dir)
        if not overall_success or not os.path.exists(os.path.join(abs_work_dir, "bragg_mask.pickle")):
            logger.error("dials.generate_mask failed or bragg_mask.pickle not created."); raise RuntimeError("DIALS Generate Mask failed")
            
        # --- Post-DIALS Python Script Steps ---
        if args.skip_post_processing:
            logger.info("Skipping post-DIALS Python script steps as requested.")
        elif overall_success: # Only proceed if DIALS steps were successful
            logger.info("--- Step 6: extract_dials_data_for_eryx.py ---")
            
            # Construct arguments for extract_dials_data_for_eryx.py
            # Paths to files created by DIALS are now relative to abs_work_dir
            # Paths to original inputs (CBF, PDB) should be absolute
            extract_cmd = [
                "python", os.path.abspath(os.path.join(orig_dir, "extract_dials_data_for_eryx.py")),
                "--experiment_file", os.path.join(abs_work_dir, final_refined_expt_name),
                "--image_files", abs_cbf_file, # Original CBF path
                "--bragg_mask_file", os.path.join(abs_work_dir, "bragg_mask.pickle"),
                "--external_pdb_file", abs_external_pdb, # Original external PDB path
                "--output_npz_file", os.path.join(abs_work_dir, f"{base_name}_diffuse_data.npz"),
                "--data-source", "pixels", # Default for eryx diffuse data
                f"--gain={args.extract_gain}",
                f"--cell_length_tol={EXTRACT_DEFAULTS['cell_length_tol']}", # Use defaults or pass from args
                f"--cell_angle_tol={EXTRACT_DEFAULTS['cell_angle_tol']}",
                f"--orient_tolerance_deg={args.extract_orient_tol_deg}",
                f"--pixel_step={EXTRACT_DEFAULTS['pixel_step']}"
            ]
            if EXTRACT_DEFAULTS["min_res"]: extract_cmd.append(f"--min_res={EXTRACT_DEFAULTS['min_res']}")
            if EXTRACT_DEFAULTS["max_res"]: extract_cmd.append(f"--max_res={EXTRACT_DEFAULTS['max_res']}")
            if EXTRACT_DEFAULTS["min_intensity"]: extract_cmd.append(f"--min_intensity={EXTRACT_DEFAULTS['min_intensity']}")
            if EXTRACT_DEFAULTS["max_intensity"]: extract_cmd.append(f"--max_intensity={EXTRACT_DEFAULTS['max_intensity']}")
            if EXTRACT_DEFAULTS["lp_correction_enabled"]: extract_cmd.append("--lp_correction_enabled")
            if EXTRACT_DEFAULTS["subtract_background_value"] is not None: extract_cmd.append(f"--subtract_background_value={EXTRACT_DEFAULTS['subtract_background_value']}")
            if EXTRACT_DEFAULTS["plot"]: extract_cmd.append("--plot")
            if EXTRACT_DEFAULTS["verbose_extract"]: extract_cmd.append("--verbose")
            
            overall_success, _ = run_command(extract_cmd, "extract_diffuse_data.log", work_dir=abs_work_dir) # Log also in work_dir
            if not overall_success:
                 logger.error("extract_dials_data_for_eryx.py failed."); raise RuntimeError("Extraction failed")

            if args.run_diagnostics and overall_success:
                logger.info("--- Step 7: calculate_q_per_pixel.py (Diagnostic) ---")
                q_map_cmd = [
                    "python", os.path.abspath(os.path.join(orig_dir, "calculate_q_per_pixel.py")),
                    "--expt", os.path.join(abs_work_dir, final_refined_expt_name), # Corrected path
                    "--output_prefix", base_name # Will be created in abs_work_dir
                ]
                # This script might need adaptation if it assumes current dir for output
                overall_success, _ = run_command(q_map_cmd, "calculate_q_per_pixel.log", work_dir=abs_work_dir)
                if not overall_success: logger.warning("calculate_q_per_pixel.py (diagnostic) failed.") # Don't make it a fatal error

                logger.info("--- Step 8: check_q_vector_consistency.py (Diagnostic) ---")
                q_check_cmd = [
                    "python", os.path.abspath(os.path.join(orig_dir, "check_q_vector_consistency.py")),
                    "--expt", os.path.join(abs_work_dir, final_refined_expt_name), # Corrected path
                    "--refl", os.path.join(abs_work_dir, final_refined_refl_name)  # Corrected path
                ]
                # This script might save plots in current dir (abs_work_dir)
                overall_success, _ = run_command(q_check_cmd, "check_q_consistency.log", work_dir=abs_work_dir)
                if not overall_success: logger.warning("check_q_vector_consistency.py (diagnostic) failed.") # Don't make it a fatal error
        
        else: # overall_success was false before post-processing
             logger.error("DIALS processing failed before post-processing steps.")
             # overall_success is already False

    except RuntimeError as e:
        logger.error(f"A critical DIALS step failed: {e}")
        overall_success = False
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        overall_success = False
    finally:
        os.chdir(orig_dir) # Always change back to the original directory
    
    if overall_success:
        logger.info(f"Processing for {args.cbf_file} completed successfully. Results in {abs_work_dir}")
        return 0
    else:
        logger.error(f"Processing for {args.cbf_file} failed. Check logs in {abs_work_dir}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
