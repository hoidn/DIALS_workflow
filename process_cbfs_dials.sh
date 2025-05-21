#!/bin/bash

# Script to process one or more CBF files with DIALS for indexing.

# --- Configuration ---
# Known unit cell and space group for lysozyme P1 (EXAMPLE - OVERRIDE IF EXTERNAL PDB USED)
# UNIT_CELL="27.424,32.134,34.513,88.66,108.46,111.88" 
# SPACE_GROUP="P1"

# Path to the PHIL file for dials.find_spots
FIND_SPOTS_PHIL_FILE="./find_spots.phil"
# Path to general DIALS indexing PHIL file (can be overridden by command line)
INDEXING_PHIL_FILE="./indexing_params.phil" 
# Path to general DIALS refinement PHIL file (can be overridden by command line)
REFINEMENT_PHIL_FILE="./refine_detector.phil" # Name suggests detector, but could be general

# --- Configuration for extract_dials_data_for_eryx.py ---
EXTRACT_MIN_RES=""
EXTRACT_MAX_RES=""
EXTRACT_MIN_INTENSITY=""
EXTRACT_MAX_INTENSITY=""
EXTRACT_GAIN="1.0"
EXTRACT_CELL_LENGTH_TOL="0.1"
EXTRACT_CELL_ANGLE_TOL="1.0"
EXTRACT_ORIENT_TOL_DEG="0.5"
EXTRACT_PIXEL_STEP="1"
EXTRACT_LP_CORRECTION_ENABLED=false  # Set to true to enable LP correction
EXTRACT_SUBTRACT_BACKGROUND_VALUE="" # Set to a number to subtract a constant background
EXTRACT_PLOT=false                  # Set to true to generate diagnostic plots
EXTRACT_VERBOSE=false               # Set to true for detailed output
RUN_DIAGNOSTICS=true # Set to false to skip q-map generation and consistency checks
# --------------------------------------------------------

MIN_SPOT_SIZE=3
AUTO_VIEW_SPOTFINDING=false
AUTO_VIEW_INDEXING_IMAGE=false
AUTO_VIEW_INDEXING_RECIP=false

# --- Script Logic ---

# Argument parsing: Expect CBF files and an optional external PDB file
# If --external_pdb is provided, it should be the last argument before CBF files.

EXTERNAL_PDB_PATH=""
CBF_FILES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --external_pdb)
      EXTERNAL_PDB_PATH="$2"
      shift # past argument
      shift # past value
      ;; 
    *.cbf)
      CBF_FILES+=("$1")
      shift # past argument
      ;;
    *)
      echo "Unknown option: $1" >&2
      # Assuming any other args might be CBF files if not starting with --
      # This is a simple parser; a more robust one would use getopts or a Python wrapper.
      if [[ "$1" != --* ]]; then 
          CBF_FILES+=("$1")
      fi
      shift
      ;;
  esac
done

# Check if external PDB is provided, error if not for eryx workflow (primary use case)
if [ -z "$EXTERNAL_PDB_PATH" ]; then
    echo ""
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "! ERROR: --external_pdb not provided.                                                 !"
    echo "! For the primary eryx data preparation workflow, an external PDB is REQUIRED         !"
    echo "! to constrain the unit cell, space group, and reference orientation.               !"
    echo "! If you intend to run general DIALS processing without PDB constraints,            !"
    echo "! please modify this script or use DIALS commands directly.                         !"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo ""
    exit 1 # Exit due to missing critical PDB for eryx workflow
fi

# Now check if the specified PDB file actually exists
if [ ! -f "$EXTERNAL_PDB_PATH" ]; then # This check was already here, just ensuring it's after the above.
    echo "Error: External PDB file not found at $EXTERNAL_PDB_PATH" >&2
    exit 1
fi


if [ ${#CBF_FILES[@]} -eq 0 ]; then
    echo "Usage: $0 --external_pdb <path_to_pdb> <cbf_file1> [cbf_file2 ...]"
    echo "Example: $0 --external_pdb ./6o2h.pdb my_image_001.cbf"
    exit 1
fi

if [ -n "$EXTERNAL_PDB_PATH" ] && [ ! -f "$EXTERNAL_PDB_PATH" ]; then
    echo "Error: External PDB file not found at $EXTERNAL_PDB_PATH" >&2
    exit 1
fi

if [ ! -f "$FIND_SPOTS_PHIL_FILE" ]; then
    echo "Error: PHIL file for spot finding not found at $FIND_SPOTS_PHIL_FILE" >&2
    exit 1
fi

START_TIME=$(date +%s)
PROCESSED_COUNT=0
FAILED_COUNT=0
ROOT_DIR=$(pwd)
LOG_SUMMARY="$ROOT_DIR/dials_processing_summary.log"

echo "DIALS Processing Summary - $(date)" > "$LOG_SUMMARY"
echo "-------------------------------------" >> "$LOG_SUMMARY"
if [ -n "$EXTERNAL_PDB_PATH" ]; then
    echo "Using External PDB: $EXTERNAL_PDB_PATH" >> "$LOG_SUMMARY"
fi

for cbf_file in "${CBF_FILES[@]}"; do
    if [ ! -f "$cbf_file" ]; then
        echo "Warning: File $cbf_file not found. Skipping."
        echo "File $cbf_file not found. Skipped." >> "$LOG_SUMMARY"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        continue
    fi

    echo "---------------------------------------------------------------------"
    echo "Processing: $cbf_file"
    echo "Processing: $cbf_file" >> "$LOG_SUMMARY"
    PROCESSED_COUNT=$((PROCESSED_COUNT + 1))

    base_name=$(basename "$cbf_file" .cbf)
    work_dir="${base_name}_dials_processing"
    mkdir -p "$work_dir"
    cd "$work_dir" || { echo "Error: Could not change to directory $work_dir. Skipping $cbf_file."; FAILED_COUNT=$((FAILED_COUNT + 1)); cd "$ROOT_DIR"; continue; }

    echo "Working directory: $(pwd)"

    DIALS_SUCCESS=true

    # 1. dials.import
    echo "Step 1: Running dials.import..."
    dials.import "../$cbf_file" output.experiments=imported.expt > dials.import.log 2>&1
    if [ $? -ne 0 ] || [ ! -f imported.expt ]; then DIALS_SUCCESS=false; echo "Error: dials.import failed. See $work_dir/dials.import.log" >> "$LOG_SUMMARY"; fi

    # 2. dials.find_spots
    if $DIALS_SUCCESS; then
        echo "Step 2: Running dials.find_spots..."
        dials.find_spots imported.expt "../$FIND_SPOTS_PHIL_FILE" spotfinder.filter.min_spot_size=$MIN_SPOT_SIZE output.reflections=strong.refl output.shoeboxes=True > dials.find_spots.log 2>&1
        if [ $? -ne 0 ] || [ ! -f strong.refl ]; then DIALS_SUCCESS=false; echo "Error: dials.find_spots failed. See $work_dir/dials.find_spots.log" >> "$LOG_SUMMARY"; 
        else
            SPOTS_FOUND=$(grep "Saved .* reflections to strong.refl" dials.find_spots.log | awk '{print $2}')
            echo "Found $SPOTS_FOUND spots." >> "$LOG_SUMMARY"
            if [ -z "$SPOTS_FOUND" ] || [ "$SPOTS_FOUND" -eq 0 ]; then echo "No spots found, cannot proceed." >> "$LOG_SUMMARY"; DIALS_SUCCESS=false; fi
        fi
    fi

    # 3. dials.index 
    INDEXING_ARGS=("imported.expt" "strong.refl" "output.experiments=indexed.expt" "output.reflections=indexed.refl")
    # Note on PHIL file precedence: Command-line arguments generally override PHIL file settings.
    # If EXTERNAL_PDB_PATH is provided, the known_symmetry.model and reference_model parameters
    # passed directly on the command line are expected to take precedence over any conflicting
    # settings in INDEXING_PHIL_FILE for cell, space group, and reference geometry.
    if [ -f "../$INDEXING_PHIL_FILE" ]; then INDEXING_ARGS+=("../$INDEXING_PHIL_FILE"); fi

    if [ -n "$EXTERNAL_PDB_PATH" ]; then
        # Forcing use of external PDB for cell/SG, and as reference geometry
        # This overrides any UNIT_CELL/SPACE_GROUP variables set in this script or PHIL files if they conflict.
        # Updated parameter names for current DIALS versions
        INDEXING_ARGS+=("indexing.known_symmetry.model=../$EXTERNAL_PDB_PATH" "indexing.reference_model.enabled=True" "indexing.reference_model.file=../$EXTERNAL_PDB_PATH")
        # Remove default unit_cell/space_group if external PDB is used for these
        # This depends on how DIALS prioritizes model vs explicit cell/sg parameters.
        # For clarity, we might only specify the model.
    else
        # Fallback to script defaults if no external PDB and PHIL doesn't specify
        # This path is no longer taken for eryx workflow as --external_pdb is mandatory.
        # Keeping for completeness or if script is adapted for general use.
        # INDEXING_ARGS+=("indexing.known_symmetry.unit_cell=$UNIT_CELL" "indexing.known_symmetry.space_group=$SPACE_GROUP")
        echo "Warning: --external_pdb not provided logic path in dials.index reached. This should not happen for eryx workflow." >> "$LOG_SUMMARY"
    fi

    if $DIALS_SUCCESS; then
        echo "Step 3: Running dials.index..."
        dials.index "${INDEXING_ARGS[@]}" > dials.index.log 2>&1
        if [ $? -ne 0 ] || [ ! -f indexed.expt ] || [ ! -f indexed.refl ]; then DIALS_SUCCESS=false; echo "Error: dials.index failed. See $work_dir/dials.index.log" >> "$LOG_SUMMARY"; fi
    fi

    # 4. dials.refine
    REFINEMENT_ARGS=("indexed.expt" "indexed.refl" "output.experiments=refined.expt" "output.reflections=refined.refl")
    if [ -f "../$REFINEMENT_PHIL_FILE" ]; then REFINEMENT_ARGS+=("../$REFINEMENT_PHIL_FILE"); fi
    
    # Phase 0, T0.5: parameterisation.crystal.fix=cell and use external PDB as reference_geometry
    if [ -n "$EXTERNAL_PDB_PATH" ]; then
        REFINEMENT_ARGS+=("refinement.parameterisation.crystal.fix=cell" "refinement.reference_model.enabled=True" "refinement.reference_model.file=../$EXTERNAL_PDB_PATH")
    fi

    if $DIALS_SUCCESS; then
        echo "Step 4: Running dials.refine..."
        dials.refine "${REFINEMENT_ARGS[@]}" > dials.refine.log 2>&1
        if [ $? -ne 0 ] || [ ! -f refined.expt ] || [ ! -f refined.refl ]; then DIALS_SUCCESS=false; echo "Error: dials.refine failed. See $work_dir/dials.refine.log" >> "$LOG_SUMMARY"; fi
    fi

    # T0.7: dials.generate_mask (Bragg mask)
    if $DIALS_SUCCESS; then
        echo "Step 5: Running dials.generate_mask..."
        # Tune parameters here as needed, e.g. border=0, d_min= (high res limit for spots to mask)
        dials.generate_mask experiments=refined.expt reflections=refined.refl output.mask=bragg_mask.pickle > dials.generate_mask.log 2>&1
        if [ $? -ne 0 ] || [ ! -f bragg_mask.pickle ]; then 
            echo "Error: dials.generate_mask failed or bragg_mask.pickle not created. Check $work_dir/dials.generate_mask.log" >> "$LOG_SUMMARY"
            DIALS_SUCCESS=false # Mark as failure as mask is crucial
        else
            echo "dials.generate_mask successful: bragg_mask.pickle" >> "$LOG_SUMMARY"
        fi
    fi

    # --- POST-DIALS PYTHON SCRIPT STEPS ---
    if $DIALS_SUCCESS && [ -f refined.expt ] && [ -f refined.refl ]; then
        echo "DIALS core processing successful. Proceeding with Python analysis scripts..."

        # Check for bragg_mask.pickle for extract_dials_data_for_eryx.py
        EXTRACTION_ARGS=("--experiment_file" "refined.expt" "--image_files" "../$cbf_file" "--output_npz_file" "${base_name}_diffuse_data.npz")
        if [ -f bragg_mask.pickle ]; then
            EXTRACTION_ARGS+=("--bragg_mask_file" "bragg_mask.pickle")
        else
            echo "Warning: bragg_mask.pickle not found. extract_dials_data_for_eryx.py might require it or run with default behavior." >> "$LOG_SUMMARY"
        fi
        if [ -n "$EXTERNAL_PDB_PATH" ]; then
             EXTRACTION_ARGS+=("--external_pdb_file" "../$EXTERNAL_PDB_PATH")
        fi

        # Add new extraction parameters conditionally
        if [ -n "$EXTRACT_MIN_RES" ]; then EXTRACTION_ARGS+=("--min_res" "$EXTRACT_MIN_RES"); fi
        if [ -n "$EXTRACT_MAX_RES" ]; then EXTRACTION_ARGS+=("--max_res" "$EXTRACT_MAX_RES"); fi
        if [ -n "$EXTRACT_MIN_INTENSITY" ]; then EXTRACTION_ARGS+=("--min_intensity" "$EXTRACT_MIN_INTENSITY"); fi
        if [ -n "$EXTRACT_MAX_INTENSITY" ]; then EXTRACTION_ARGS+=("--max_intensity" "$EXTRACT_MAX_INTENSITY"); fi
        EXTRACTION_ARGS+=("--gain" "$EXTRACT_GAIN")
        EXTRACTION_ARGS+=("--cell_length_tol" "$EXTRACT_CELL_LENGTH_TOL")
        EXTRACTION_ARGS+=("--cell_angle_tol" "$EXTRACT_CELL_ANGLE_TOL")
        EXTRACTION_ARGS+=("--orient_tolerance_deg" "$EXTRACT_ORIENT_TOL_DEG")
        
        echo "Running extract_dials_data_for_eryx.py (for diffuse data)..."
        python "$ROOT_DIR/extract_dials_data_for_eryx.py" "${EXTRACTION_ARGS[@]}" > extract_diffuse_data.log 2>&1
        EXTRACTION_EXIT_CODE=$?
        if [ $EXTRACTION_EXIT_CODE -ne 0 ]; then 
            echo "Error: extract_dials_data_for_eryx.py (diffuse) failed with exit code $EXTRACTION_EXIT_CODE. See log." >> "$LOG_SUMMARY"
            DIALS_SUCCESS=false # Consider this a failure of the main eryx prep path
        else 
            echo "extract_dials_data_for_eryx.py (diffuse) successful." >> "$LOG_SUMMARY"
        fi
        
        if [ "$RUN_DIAGNOSTICS" = true ] && $DIALS_SUCCESS; then # Only run diagnostics if main extraction succeeded
            echo "Running diagnostic scripts..."
            # Q-map generation (calculate_q_per_pixel.py)
            echo "Running calculate_q_per_pixel.py..."
            python "$ROOT_DIR/calculate_q_per_pixel.py" --expt refined.expt --output_prefix "${base_name}" > calculate_q_per_pixel.log 2>&1
            if [ $? -ne 0 ]; then echo "calculate_q_per_pixel.py failed. See log." >> "$LOG_SUMMARY"; else echo "calculate_q_per_pixel.py successful." >> "$LOG_SUMMARY"; fi

            # Q-vector consistency check (check_q_vector_consistency.py)
            echo "Running check_q_vector_consistency.py..."
            python "$ROOT_DIR/check_q_vector_consistency.py" --expt refined.expt --refl refined.refl > check_q_consistency.log 2>&1
            if [ $? -ne 0 ]; then echo "check_q_vector_consistency.py failed. See log." >> "$LOG_SUMMARY"; else echo "check_q_vector_consistency.py successful." >> "$LOG_SUMMARY"; fi
        else
            echo "Skipping diagnostic scripts (RUN_DIAGNOSTICS is not true)." >> "$LOG_SUMMARY"
        fi

    else
        echo "DIALS core processing failed or essential files missing. Skipping Python analysis scripts." >> "$LOG_SUMMARY"
        if $DIALS_SUCCESS; then FAILED_COUNT=$((FAILED_COUNT + 1)); fi # If DIALS_SUCCESS true but files missing
    fi
    
    if ! $DIALS_SUCCESS; then FAILED_COUNT=$((FAILED_COUNT+1)); fi
    cd "$ROOT_DIR"
    echo "Finished processing $cbf_file."
    echo ""
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "---------------------------------------------------------------------"
echo "DIALS Processing Complete."
echo "Total CBF files attempted: $PROCESSED_COUNT"
echo "Failed DIALS steps or skipped Python analysis: $FAILED_COUNT"
# A more nuanced success count could be added here based on log grepping
TOTAL_FULLY_SUCCESSFUL=$(grep -c "extract_dials_data_for_eryx.py (diffuse) successful." "$LOG_SUMMARY")
echo "Fully successful (DIALS + diffuse extraction): $TOTAL_FULLY_SUCCESSFUL"

echo "Total processing time: $DURATION seconds."
echo "Summary log created: $LOG_SUMMARY"

echo "-------------------------------------" >> "$LOG_SUMMARY"
echo "Total CBF files attempted: $PROCESSED_COUNT" >> "$LOG_SUMMARY"
echo "Failed DIALS steps or skipped Python analysis: $FAILED_COUNT" >> "$LOG_SUMMARY"
echo "Fully successful (DIALS + diffuse extraction): $TOTAL_FULLY_SUCCESSFUL" >> "$LOG_SUMMARY"
echo "Total processing time: $DURATION seconds." >> "$LOG_SUMMARY"
