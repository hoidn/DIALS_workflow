#!/bin/bash

# Script to process one or more CBF files with DIALS for indexing.

# --- Configuration ---
# Known unit cell and space group for lysozyme P1
UNIT_CELL="27.424,32.134,34.513,88.66,108.46,111.88"
SPACE_GROUP="P1"

# Path to the PHIL file for dials.find_spots
# Assumes find_spots.phil is in the same directory as this script.
# If not, provide the full path.
FIND_SPOTS_PHIL_FILE="./find_spots.phil"
# Example: FIND_SPOTS_PHIL_FILE="/path/to/your/find_spots.phil"

# Spot finding parameters (these are on the command line, PHIL file handles threshold)
MIN_SPOT_SIZE=3

# Set to true to automatically open viewers, false to skip
AUTO_VIEW_SPOTFINDING=false
AUTO_VIEW_INDEXING_IMAGE=false
AUTO_VIEW_INDEXING_RECIP=false

# --- Script Logic ---

# Check if CBF files are provided
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <cbf_file1> [cbf_file2 ...]"
    echo "Example: $0 my_image_001.cbf my_image_002.cbf"
    exit 1
fi

# Check if find_spots.phil exists
if [ ! -f "$FIND_SPOTS_PHIL_FILE" ]; then
    echo "Error: PHIL file for spot finding not found at $FIND_SPOTS_PHIL_FILE"
    echo "Please create it with the following content or update the path in the script:"
    echo ""
    echo "spotfinder {"
    echo "  threshold {"
    echo "    algorithm = dispersion"
    echo "    dispersion {"
    echo "      sigma_strong = 3"
    echo "    }"
    echo "  }"
    echo "}"
    exit 1
fi

START_TIME=$(date +%s)
PROCESSED_COUNT=0
FAILED_COUNT=0
LOG_SUMMARY="dials_processing_summary.log"

echo "DIALS Processing Summary - $(date)" > "$LOG_SUMMARY"
echo "-------------------------------------" >> "$LOG_SUMMARY"

# Loop through each CBF file provided as an argument
for cbf_file in "$@"; do
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

    # Create a unique working directory for this CBF file
    base_name=$(basename "$cbf_file" .cbf)
    work_dir="${base_name}_dials_processing"
    mkdir -p "$work_dir"
    cd "$work_dir" || { echo "Error: Could not change to directory $work_dir. Skipping $cbf_file."; FAILED_COUNT=$((FAILED_COUNT + 1)); cd ..; continue; }

    echo "Working directory: $(pwd)"

    # 1. dials.import
    echo "Step 1: Running dials.import..."
    dials.import "../$cbf_file" output.experiments=imported.expt > dials.import.log 2>&1
    if [ $? -ne 0 ] || [ ! -f imported.expt ]; then
        echo "Error: dials.import failed for $cbf_file. Check dials.import.log in $work_dir"
        echo "dials.import failed for $cbf_file" >> "$LOG_SUMMARY"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        cd ..
        continue
    fi
    echo "dials.import successful."

    # 2. dials.find_spots
    echo "Step 2: Running dials.find_spots..."
    dials.find_spots imported.expt \
      "../$FIND_SPOTS_PHIL_FILE" \
      spotfinder.filter.min_spot_size="$MIN_SPOT_SIZE" \
      output.reflections=strong.refl \
      output.shoeboxes=True > dials.find_spots.log 2>&1 # Set to False if shoeboxes not needed for viewing
    if [ $? -ne 0 ] || [ ! -f strong.refl ]; then
        echo "Error: dials.find_spots failed for $cbf_file. Check dials.find_spots.log in $work_dir"
        echo "dials.find_spots failed for $cbf_file" >> "$LOG_SUMMARY"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        cd ..
        continue
    fi
    # Get number of spots found
    SPOTS_FOUND=$(grep "Saved .* reflections to strong.refl" dials.find_spots.log | awk '{print $2}')
    echo "dials.find_spots successful. Found $SPOTS_FOUND spots."
    echo "Found $SPOTS_FOUND spots for $cbf_file." >> "$LOG_SUMMARY"

    if [ "$AUTO_VIEW_SPOTFINDING" = true ] && [ -n "$SPOTS_FOUND" ] && [ "$SPOTS_FOUND" -ne 0 ]; then
        echo "Opening spot finding viewer (close to continue)..."
        dials.image_viewer imported.expt strong.refl
    fi

    if [ -z "$SPOTS_FOUND" ] || [ "$SPOTS_FOUND" -eq 0 ]; then
        echo "Warning: No spots found by dials.find_spots for $cbf_file. Skipping indexing."
        echo "No spots found for $cbf_file. Skipped indexing." >> "$LOG_SUMMARY"
        # No need to increment FAILED_COUNT here if find_spots itself didn't error,
        # but it's a processing dead-end for this file.
        cd ..
        continue
    fi

    # 3. dials.index (initial indexing, detector likely fixed by default)
    echo "Step 3: Running dials.index (initial, detector likely fixed)..."
    dials.index imported.expt strong.refl \
      indexing.known_symmetry.unit_cell="$UNIT_CELL" \
      indexing.known_symmetry.space_group="$SPACE_GROUP" \
      output.experiments=indexed_initial.expt \
      output.reflections=indexed_initial.refl > dials.index.log 2>&1
    
    if [ $? -ne 0 ] || [ ! -f indexed_initial.expt ] || [ ! -f indexed_initial.refl ]; then
        echo "Error: dials.index (initial) failed for $cbf_file. Check dials.index.log in $work_dir"
        echo "dials.index (initial) failed for $cbf_file" >> "$LOG_SUMMARY"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        cd ..
        continue
    fi
    # Get number of indexed reflections
    INDEXED_COUNT=$(grep -A1 "Refined crystal models:" dials.index.log | tail -n1 | awk -F'(' '{print $2}' | awk '{print $1}')
    # Alternative grep for final summary table:
    if [ -z "$INDEXED_COUNT" ]; then
        INDEXED_COUNT=$(grep -A3 "Saving refined experiments to indexed_initial.expt" dials.index.log | grep "% indexed" | awk '{print $3}')
    fi
    PERCENT_INDEXED=$(grep -A3 "Saving refined experiments to indexed_initial.expt" dials.index.log | grep "% indexed" | awk '{print $9}')

    echo "dials.index (initial) successful. Indexed $INDEXED_COUNT reflections ($PERCENT_INDEXED %)."
    echo "Indexed $INDEXED_COUNT ($PERCENT_INDEXED %) for $cbf_file (initial)." >> "$LOG_SUMMARY"

    # 4. dials.refine (dedicated refinement of detector and beam geometry using PHIL)
    echo "Step 4: Running dials.refine (for detector and beam geometry using PHIL)..."
    dials.refine indexed_initial.expt indexed_initial.refl \
      "../refine_detector.phil" \ # Correct path to your PHIL file
      output.experiments=indexed_refined_detector.expt \
      output.reflections=indexed_refined_detector.refl \
      > dials.refine.log 2>&1

    if [ $? -ne 0 ] || [ ! -f indexed_refined_detector.expt ] || [ ! -f indexed_refined_detector.refl ]; then
        echo "Error: dials.refine failed for $cbf_file. Check dials.refine.log in $work_dir"
        echo "dials.refine failed for $cbf_file" >> "$LOG_SUMMARY"
        FAILED_COUNT=$((FAILED_COUNT + 1)) 
        cd ..
        continue
    fi
    echo "dials.refine successful."

    # Update any subsequent automatic viewing commands to use 
    # indexed_refined_detector.expt and indexed_refined_detector.refl
    if [ "$AUTO_VIEW_INDEXING_IMAGE" = true ] && [ -n "$INDEXED_COUNT" ] && [ "$INDEXED_COUNT" -ne 0 ]; then
        echo "Opening refined indexing image viewer (close to continue)..."
        dials.image_viewer indexed_refined_detector.expt indexed_refined_detector.refl
    fi
    if [ "$AUTO_VIEW_INDEXING_RECIP" = true ] && [ -n "$INDEXED_COUNT" ] && [ "$INDEXED_COUNT" -ne 0 ]; then
        echo "Opening refined reciprocal lattice viewer (close to continue)..."
        dials.reciprocal_lattice_viewer indexed_refined_detector.expt indexed_refined_detector.refl
    fi

    # Go back to the parent directory for the next file
    cd ..
    echo "Finished processing $cbf_file."
    echo ""
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "---------------------------------------------------------------------"
echo "DIALS Processing Complete."
echo "Total CBF files attempted: $PROCESSED_COUNT"
echo "Successfully processed (indexed): $((PROCESSED_COUNT - FAILED_COUNT))" # This is a simplification
echo "Failed or skipped: $FAILED_COUNT"
echo "Total processing time: $DURATION seconds."
echo "Summary log created: $LOG_SUMMARY"

echo "-------------------------------------" >> "$LOG_SUMMARY"
echo "Total CBF files attempted: $PROCESSED_COUNT" >> "$LOG_SUMMARY"
echo "Failed or skipped steps: $FAILED_COUNT" >> "$LOG_SUMMARY"
echo "Total processing time: $DURATION seconds." >> "$LOG_SUMMARY"
