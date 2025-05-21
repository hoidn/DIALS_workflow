#!/bin/bash

# Script to process CBF files using a simplified DIALS workflow (like the "old working" version)
# and then call Python scripts for data extraction and consistency checks.
# External PDB is used ONLY by the Python script for consistency checks.
# DIALS uses hardcoded unit cell and space group, with the cell fixed during refinement.

# --- Configuration ---
# HARDCODED Unit cell and space group - MUST MATCH THE PDB USED FOR ERYX/CONSISTENCY CHECKS
UNIT_CELL="27.424,32.134,34.513,88.66,108.46,111.88" # Example for 6o2h.pdb (Triclinic Lysozyme P1)
SPACE_GROUP="P1"                                     # Example for 6o2h.pdb

# Path to PHIL files (assumed to be in the same directory as this script or CWD)
FIND_SPOTS_PHIL_FILE="./find_spots.phil"
# General PHIL for refinement - ensure this does NOT set reference_model or try to change cell if fix=cell is used.
# It's primarily for detector/beam refinement parameters.
REFINEMENT_PHIL_FILE="./refine_detector.phil" # Renamed for clarity

# DIALS parameters
MIN_SPOT_SIZE=3

# --- Configuration for Python Scripts ---
EXTERNAL_PDB_FOR_ERYX_CHECK="" # To be set by --external_pdb argument

# Filtering and processing parameters for the extraction script
EXTRACT_MIN_RES="50.0" # d_max in Angstrom
EXTRACT_MAX_RES="1.5"  # d_min in Angstrom
EXTRACT_MIN_INTENSITY="0" # Minimum pixel intensity for diffuse data
EXTRACT_MAX_INTENSITY="60000" # Maximum pixel intensity (saturation)
EXTRACT_GAIN="1.0"
EXTRACT_CELL_LENGTH_TOL="0.01" # Tolerance for Python script's cell check
EXTRACT_CELL_ANGLE_TOL="0.1"  # Tolerance for Python script's cell check
EXTRACT_ORIENT_TOL_DEG="1.0"  # Orientation tolerance vs external PDB in Python script
EXTRACT_PIXEL_STEP="1"        # Process every Nth pixel in Python script
EXTRACT_LP_CORRECTION_ENABLED=false # Enable simplified LP correction in Python script
EXTRACT_SUBTRACT_BACKGROUND_VALUE="" # Constant value to subtract from pixels in Python script
EXTRACT_PLOT=false            # Generate diagnostic plots from Python extraction script
EXTRACT_VERBOSE_PYTHON=false  # Verbose output from Python scripts

# Diagnostic script execution
RUN_DIAGNOSTICS=true # Set to false to skip calculate_q_per_pixel.py and check_q_vector_consistency.py
# --------------------------------------------------------

# --- Script Logic ---
CBF_FILES=()
# Argument parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --external_pdb)
      EXTERNAL_PDB_FOR_ERYX_CHECK="$2"
      shift 2
      ;;
    --run_diagnostics)
      if [[ "$2" == "true" || "$2" == "false" ]]; then RUN_DIAGNOSTICS="$2"; shift 2;
      else echo "Error: --run_diagnostics requires true or false." >&2; exit 1; fi
      ;;
    --verbose) EXTRACT_VERBOSE_PYTHON=true; shift ;;
    *.cbf) CBF_FILES+=("$1"); shift ;;
    *) if [[ "$1" != --* ]]; then CBF_FILES+=("$1"); else echo "Unknown option: $1" >&2; fi; shift ;;
  esac
done

# Mandatory check for external PDB (for Python script's consistency check)
if [ -z "$EXTERNAL_PDB_FOR_ERYX_CHECK" ]; then echo "ERROR: --external_pdb <path> is REQUIRED for Python script consistency checks." >&2; exit 1; fi
if [ ! -f "$EXTERNAL_PDB_FOR_ERYX_CHECK" ]; then echo "Error: External PDB for eryx check not found at $EXTERNAL_PDB_FOR_ERYX_CHECK" >&2; exit 1; fi
if [ ${#CBF_FILES[@]} -eq 0 ]; then echo "Usage: $0 --external_pdb <path_to_pdb_for_eryx_check> <cbf_file1> [cbf_file2 ...]"; exit 1; fi
if [ ! -f "$FIND_SPOTS_PHIL_FILE" ]; then echo "Error: Spot finding PHIL not found at $FIND_SPOTS_PHIL_FILE" >&2; exit 1; fi


START_TIME=$(date +%s)
PROCESSED_COUNT=0
FAILED_DIALS_STEPS=0
FAILED_EXTRACTION_STEPS=0
ROOT_DIR=$(pwd) # Directory where the script is launched
LOG_SUMMARY="$ROOT_DIR/dials_processing_summary.log"

# Initialize Summary Log
echo "DIALS & Extraction Processing Summary - $(date)" > "$LOG_SUMMARY"
echo "-------------------------------------------------" >> "$LOG_SUMMARY"
echo "Script Root Directory: $ROOT_DIR" >> "$LOG_SUMMARY"
echo "Using DIALS Unit Cell (hardcoded): $UNIT_CELL" >> "$LOG_SUMMARY"
echo "Using DIALS Space Group (hardcoded): $SPACE_GROUP" >> "$LOG_SUMMARY"
echo "External PDB for Python script consistency check: $EXTERNAL_PDB_FOR_ERYX_CHECK" >> "$LOG_SUMMARY"
echo "Diagnostic scripts will run: $RUN_DIAGNOSTICS" >> "$LOG_SUMMARY"
echo "Python script verbosity: $EXTRACT_VERBOSE_PYTHON" >> "$LOG_SUMMARY"
echo "-------------------------------------------------" >> "$LOG_SUMMARY"


for cbf_file_rel_path in "${CBF_FILES[@]}"; do
    cbf_file_abs_path=$(realpath "$cbf_file_rel_path") 

    if [ ! -f "$cbf_file_abs_path" ]; then
        echo "Warning: CBF file $cbf_file_rel_path (abs: $cbf_file_abs_path) not found. Skipping." | tee -a "$LOG_SUMMARY"
        FAILED_DIALS_STEPS=$((FAILED_DIALS_STEPS + 1))
        continue
    fi

    echo ""
    echo "---------------------------------------------------------------------"
    echo "Processing: $cbf_file_rel_path (abs: $cbf_file_abs_path)"
    echo "Processing: $cbf_file_rel_path" >> "$LOG_SUMMARY"
    PROCESSED_COUNT=$((PROCESSED_COUNT + 1))

    base_name=$(basename "$cbf_file_rel_path" .cbf)
    work_dir="$ROOT_DIR/${base_name}_dials_processing"
    mkdir -p "$work_dir"
    
    pushd . > /dev/null # Save current directory
    cd "$work_dir" || { echo "Error: Could not change to directory $work_dir. Skipping $cbf_file_rel_path."; FAILED_DIALS_STEPS=$((FAILED_DIALS_STEPS + 1)); popd > /dev/null; continue; }

    echo "Working directory: $(pwd)"
    CURRENT_FILE_DIALS_SUCCESS=true

    # === DIALS STEPS (Simplified calls, similar to "old working" script) ===
    # Output names for intermediate DIALS files
    IMPORTED_EXPT="imported.expt"
    STRONG_REFL="strong.refl"
    INDEXED_EXPT="indexed.expt" # Changed from indexed_initial for simplicity
    INDEXED_REFL="indexed.refl" # Changed from indexed_initial
    REFINED_EXPT="refined.expt" # Final refined experiment for Python scripts
    REFINED_REFL="refined.refl" # Final refined reflections
    BRAGG_MASK="bragg_mask.pickle"

    # 1. dials.import
    echo "Step 1: Running dials.import..."
    dials.import "$cbf_file_abs_path" output.experiments="$IMPORTED_EXPT" > dials.import.log 2>&1
    if [ $? -ne 0 ] || [ ! -s "$IMPORTED_EXPT" ]; then 
        CURRENT_FILE_DIALS_SUCCESS=false
        echo "Error: dials.import failed for $cbf_file_rel_path. Check log in $(pwd)" >> "$LOG_SUMMARY"
    fi

    # 2. dials.find_spots
    if $CURRENT_FILE_DIALS_SUCCESS; then
        echo "Step 2: Running dials.find_spots..."
        dials.find_spots "$IMPORTED_EXPT" \
          "$ROOT_DIR/$FIND_SPOTS_PHIL_FILE" \
          spotfinder.filter.min_spot_size="$MIN_SPOT_SIZE" \
          output.reflections="$STRONG_REFL" \
          output.shoeboxes=True > dials.find_spots.log 2>&1
        if [ $? -ne 0 ] || [ ! -s "$STRONG_REFL" ]; then
            CURRENT_FILE_DIALS_SUCCESS=false
            echo "Error: dials.find_spots failed for $cbf_file_rel_path. Check log." >> "$LOG_SUMMARY"
        else
            SPOTS_FOUND=$(grep "Saved .* reflections to $STRONG_REFL" dials.find_spots.log | awk '{print $2}')
            echo "Found $SPOTS_FOUND spots for $cbf_file_rel_path." >> "$LOG_SUMMARY"
            if [ -z "$SPOTS_FOUND" ] || [ "$SPOTS_FOUND" -eq 0 ]; then
                echo "Warning: No spots found by dials.find_spots for $cbf_file_rel_path. Cannot proceed." >> "$LOG_SUMMARY"
                CURRENT_FILE_DIALS_SUCCESS=false
            fi
        fi
    fi

    # 3. dials.index (using hardcoded cell/SG)
    if $CURRENT_FILE_DIALS_SUCCESS; then
        echo "Step 3: Running dials.index..."
        INDEXING_COMMAND_ARGS=("$IMPORTED_EXPT" "$STRONG_REFL")
        INDEXING_COMMAND_ARGS+=("indexing.known_symmetry.unit_cell=$UNIT_CELL")
        INDEXING_COMMAND_ARGS+=("indexing.known_symmetry.space_group=$SPACE_GROUP")
        # Add any other essential indexing parameters directly
        INDEXING_COMMAND_ARGS+=("indexing.stills.indexer=stills" "indexing.max_cell=500") # Example
        INDEXING_COMMAND_ARGS+=("output.experiments=$INDEXED_EXPT" "output.reflections=$INDEXED_REFL")
        
        dials.index "${INDEXING_COMMAND_ARGS[@]}" > dials.index.log 2>&1
        if [ $? -ne 0 ] || [ ! -s "$INDEXED_EXPT" ] || [ ! -s "$INDEXED_REFL" ]; then
            CURRENT_FILE_DIALS_SUCCESS=false
            echo "Error: dials.index failed for $cbf_file_rel_path. Check log." >> "$LOG_SUMMARY"
        fi
    fi
    
    # 4. dials.refine
    if $CURRENT_FILE_DIALS_SUCCESS; then
        echo "Step 4: Running dials.refine..."
        REFINE_COMMAND_ARGS=("$INDEXED_EXPT" "$INDEXED_REFL")
        # CRITICAL: Ensure unit cell is fixed. This overrides PHIL if PHIL tries to unfix it.
        REFINE_COMMAND_ARGS+=("refinement.parameterisation.crystal.fix=cell")
        if [ -f "$ROOT_DIR/$REFINEMENT_PHIL_FILE" ]; then
            REFINE_COMMAND_ARGS+=("$ROOT_DIR/$REFINEMENT_PHIL_FILE")
        else
            echo "Warning: Refinement PHIL file $ROOT_DIR/$REFINEMENT_PHIL_FILE not found. Running refine with minimal command line args." >> "$LOG_SUMMARY"
            # Add essential refinement parameters if PHIL is missing, e.g., for outlier rejection
            # REFINE_COMMAND_ARGS+=("refinement.reflections.outlier.algorithm=tukey")
        fi
        REFINE_COMMAND_ARGS+=("output.experiments=$REFINED_EXPT" "output.reflections=$REFINED_REFL")

        dials.refine "${REFINE_COMMAND_ARGS[@]}" > dials.refine.log 2>&1
        if [ $? -ne 0 ] || [ ! -s "$REFINED_EXPT" ] || [ ! -s "$REFINED_REFL" ]; then
            CURRENT_FILE_DIALS_SUCCESS=false
            echo "Error: dials.refine failed for $cbf_file_rel_path. Check log." >> "$LOG_SUMMARY"
        fi
    fi

    # 5. dials.generate_mask (using the final refined files)
    if $CURRENT_FILE_DIALS_SUCCESS; then
        echo "Step 5: Running dials.generate_mask..."
        dials.generate_mask experiments="$REFINED_EXPT" reflections="$REFINED_REFL" output.mask="$BRAGG_MASK" > dials.generate_mask.log 2>&1
        if [ $? -ne 0 ] || [ ! -s "$BRAGG_MASK" ]; then
            CURRENT_FILE_DIALS_SUCCESS=false # Mask is crucial
            echo "Error: dials.generate_mask failed for $cbf_file_rel_path. Check log." >> "$LOG_SUMMARY"
        fi
    fi
    # === END OF DIALS STEPS ===

    if $CURRENT_FILE_DIALS_SUCCESS; then
        echo "DIALS core processing successful for $cbf_file_rel_path. Proceeding with Python scripts..." | tee -a "$LOG_SUMMARY"

        # --- POST-DIALS PYTHON SCRIPT STEPS ---
        EXTRACTION_ARGS=("--experiment_file" "$(pwd)/$REFINED_EXPT" \
                         "--image_files" "$cbf_file_abs_path" \
                         "--bragg_mask_file" "$(pwd)/$BRAGG_MASK" \
                         "--external_pdb_file" "$(realpath "$EXTERNAL_PDB_FOR_ERYX_CHECK")" \
                         "--output_npz_file" "$(pwd)/${base_name}_diffuse_data.npz" \
                         "--gain" "$EXTRACT_GAIN" \
                         "--cell_length_tol" "$EXTRACT_CELL_LENGTH_TOL" \
                         "--cell_angle_tol" "$EXTRACT_CELL_ANGLE_TOL" \
                         "--orient_tolerance_deg" "$EXTRACT_ORIENT_TOL_DEG" \
                         "--pixel_step" "$EXTRACT_PIXEL_STEP")
        if [ -n "$EXTRACT_MIN_RES" ]; then EXTRACTION_ARGS+=("--min_res" "$EXTRACT_MIN_RES"); fi
        if [ -n "$EXTRACT_MAX_RES" ]; then EXTRACTION_ARGS+=("--max_res" "$EXTRACT_MAX_RES"); fi
        if [ -n "$EXTRACT_MIN_INTENSITY" ]; then EXTRACTION_ARGS+=("--min_intensity" "$EXTRACT_MIN_INTENSITY"); fi
        if [ -n "$EXTRACT_MAX_INTENSITY" ]; then EXTRACTION_ARGS+=("--max_intensity" "$EXTRACT_MAX_INTENSITY"); fi
        if [ "$EXTRACT_LP_CORRECTION_ENABLED" = true ]; then EXTRACTION_ARGS+=("--lp_correction_enabled"); fi
        if [ -n "$EXTRACT_SUBTRACT_BACKGROUND_VALUE" ]; then EXTRACTION_ARGS+=("--subtract_background_value" "$EXTRACT_SUBTRACT_BACKGROUND_VALUE"); fi
        if [ "$EXTRACT_PLOT" = true ]; then EXTRACTION_ARGS+=("--plot"); fi
        if [ "$EXTRACT_VERBOSE_PYTHON" = true ]; then EXTRACTION_ARGS+=("--verbose"); fi
        # Default to pixel mode for eryx diffuse data
        EXTRACTION_ARGS+=("--data-source" "pixels")


        echo "Running extract_dials_data_for_eryx.py..."
        python "$ROOT_DIR/extract_dials_data_for_eryx.py" "${EXTRACTION_ARGS[@]}" > extract_diffuse_data.log 2>&1
        EXTRACTION_EXIT_CODE=$?
        if [ $EXTRACTION_EXIT_CODE -ne 0 ]; then
            echo "Error: extract_dials_data_for_eryx.py failed for $cbf_file_rel_path with exit code $EXTRACTION_EXIT_CODE. See log in $(pwd)" >> "$LOG_SUMMARY"
            FAILED_EXTRACTION_STEPS=$((FAILED_EXTRACTION_STEPS + 1))
        else
            echo "extract_dials_data_for_eryx.py successful for $cbf_file_rel_path." >> "$LOG_SUMMARY"
        fi

        if [ "$RUN_DIAGNOSTICS" = true ]; then
            echo "Running diagnostic scripts for $cbf_file_rel_path..."
            # check_q_vector_consistency.py uses refined.expt and refined.refl
            python "$ROOT_DIR/check_q_vector_consistency.py" --expt "$(pwd)/$REFINED_EXPT" --refl "$(pwd)/$REFINED_REFL" > check_q_consistency.log 2>&1
            if [ $? -ne 0 ]; then echo "Warning: check_q_vector_consistency.py failed for $cbf_file_rel_path." >> "$LOG_SUMMARY"; fi
            
            # calculate_q_per_pixel.py uses refined.expt
            python "$ROOT_DIR/calculate_q_per_pixel.py" --expt "$(pwd)/$REFINED_EXPT" --output_prefix "${base_name}" > calculate_q_per_pixel.log 2>&1
            if [ $? -ne 0 ]; then echo "Warning: calculate_q_per_pixel.py failed for $cbf_file_rel_path." >> "$LOG_SUMMARY"; fi
        fi
    else
        echo "DIALS core processing failed for $cbf_file_rel_path. Skipping Python scripts." >> "$LOG_SUMMARY"
        FAILED_DIALS_STEPS=$((FAILED_DIALS_STEPS + 1))
    fi
    
    popd > /dev/null # Return to ROOT_DIR
    echo "Finished processing $cbf_file_rel_path."
    echo ""
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
TOTAL_FULLY_SUCCESSFUL=$(grep -c "extract_dials_data_for_eryx.py successful for" "$LOG_SUMMARY") # More specific grep

echo "---------------------------------------------------------------------"
echo "DIALS & Extraction Processing Complete."
echo "Total CBF files attempted: $PROCESSED_COUNT"
echo "Failed DIALS steps (preventing extraction): $FAILED_DIALS_STEPS"
echo "Failed Extraction steps (after successful DIALS): $FAILED_EXTRACTION_STEPS"
echo "Fully successful (DIALS + diffuse extraction): $TOTAL_FULLY_SUCCESSFUL"
echo "Total processing time: $DURATION seconds."
echo "Summary log created: $LOG_SUMMARY"

echo "-------------------------------------" >> "$LOG_SUMMARY"
echo "Total CBF files attempted: $PROCESSED_COUNT" >> "$LOG_SUMMARY"
echo "Failed DIALS steps: $FAILED_DIALS_STEPS" >> "$LOG_SUMMARY"
echo "Failed Extraction steps: $FAILED_EXTRACTION_STEPS" >> "$LOG_SUMMARY"
echo "Fully successful (DIALS + diffuse extraction): $TOTAL_FULLY_SUCCESSFUL" >> "$LOG_SUMMARY"
echo "Total processing time: $DURATION seconds." >> "$LOG_SUMMARY"
