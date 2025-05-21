from __future__ import annotations

import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("dials_script")

try:
    from dxtbx.model.experiment_list import ExperimentListFactory
    from dials.array_family import flex
    from dials.algorithms.indexing.indexer import Indexer
    import dials.algorithms.indexing # To access its phil_scope for include
    import iotbx.phil
except ImportError as e:
    print(f"Error importing DIALS/CCTBX modules: {e}")
    print("Please ensure your CCTBX/DIALS environment is correctly set up and activated.")
    sys.exit(1)

# Define the master PHIL scope.
# dials.algorithms.indexing.phil_scope (from indexer.py) already starts with "indexing { ... }"
master_phil_scope_str = """
output {
  experiments = indexed_script.expt
    .type = path
  reflections = indexed_script.refl
    .type = path
  log = dials.index_script.log
    .type = path
}
# This should result in params.output and params.indexing 
# (where params.indexing contains stills, method etc. from the included scope)
include scope dials.algorithms.indexing.phil_scope
"""

def run_indexing(experiments_path: str, reflections_path: str, phil_params_str: str | None = None):
    try:
        logger.info(f"Loading experiments from: {experiments_path}")
        experiments = ExperimentListFactory.from_json_file(experiments_path, check_format=False)
        if not experiments: 
            logger.error("Failed to load experiments.")
            return
        logger.info(f"Loading reflections from: {reflections_path}")
        reflections = flex.reflection_table.from_file(reflections_path)
        if not reflections: 
            logger.error("Failed to load reflections.")
            return
        logger.info(f"Loaded {len(experiments)} experiments and {len(reflections)} reflections.")

        base_phil = iotbx.phil.parse(master_phil_scope_str)

        if phil_params_str:
            # User PHIL string should use 'indexing.' prefix for indexing params,
            # e.g., "indexing.known_symmetry.unit_cell='...'"
            user_phil = iotbx.phil.parse(phil_params_str)
            working_phil = base_phil.fetch(source=user_phil)
        else:
            working_phil = base_phil

        params = working_phil.extract() 
        
        logger.info("Effective PHIL parameters being used (working_phil):")
        logger.info(working_phil.as_str(attributes_level=2)) # Increased attributes_level for more detail

        # --- Debugging extracted params object ---
        logger.info("--- Debugging extracted params object (our script's 'params') ---")
        logger.info(f"type(params): {type(params)}")
        logger.info(f"dir(params) attributes: {dir(params)}") # Show all attributes
        
        if hasattr(params, 'output'):
            logger.info("params HAS 'output' attribute.")
        else:
            logger.info("params DOES NOT HAVE 'output' attribute.")

        if hasattr(params, 'indexing'):
            logger.info("params HAS 'indexing' attribute.")
            logger.info(f"  type(params.indexing): {type(params.indexing)}")
            logger.info(f"  dir(params.indexing) attributes: {dir(params.indexing)}")
            if hasattr(params.indexing, 'stills'):
                logger.info("  params.indexing HAS 'stills' attribute.")
                if hasattr(params.indexing.stills, 'indexer'):
                    logger.info("    params.indexing.stills HAS 'indexer' attribute.")
                else:
                    logger.info("    params.indexing.stills DOES NOT HAVE 'indexer' attribute.")
            else:
                logger.info("  params.indexing DOES NOT HAVE 'stills' attribute.")
        else:
            logger.info("params DOES NOT HAVE 'indexing' attribute.")
        logger.info("--- End Debugging ---")


        logger.info("Starting indexing process...")
        # Pass the whole 'params' object, Indexer.from_parameters will look for params.indexing internally
        logger.info("Creating indexer instance...")
        indexer = Indexer.from_parameters(
            reflections,
            experiments,
            params=params # This 'params' is our script's 'params' object
        )
        logger.info("Running indexer.index()...")
        indexer.index()

        indexed_experiments = indexer.refined_experiments
        indexed_reflections = indexer.refined_reflections 

        num_indexed = indexed_reflections.get_flags(indexed_reflections.flags.indexed).count(True)
        logger.info(f"Indexing complete. Found {num_indexed} indexed reflections.")
        
        # Print additional crystal info for debugging/validation
        if indexed_experiments and len(indexed_experiments) > 0 and indexed_experiments[0].crystal:
            crystal = indexed_experiments[0].crystal
            unit_cell = crystal.get_unit_cell()
            space_group = crystal.get_space_group()
            logger.info(f"Indexed crystal unit cell: {unit_cell.parameters()}")
            logger.info(f"Indexed crystal space group: {space_group.info().symbol_and_number()}")
            logger.info(f"Crystal A matrix: {crystal.get_A()}")

        logger.info(f"Saving indexed experiments to: {params.output.experiments}")
        indexed_experiments.as_json(params.output.experiments)
        
        logger.info(f"Saving indexed reflections to: {params.output.reflections}")
        indexed_reflections.as_file(params.output.reflections)

        logger.info(f"DIALS indexing script finished successfully.")

    except Exception as e:
        logger.error(f"An error occurred during indexing: {e}", exc_info=True)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_dials_index_programmatic.py <imported.expt_path> <strong.refl_path> [\"<phil_params_string>\"]")
        print("Example PHIL string (parameters under 'indexing.' scope):")
        print("  \"indexing.known_symmetry.unit_cell='27.4,32.1,34.5,88.6,108.4,111.8' indexing.known_symmetry.space_group=P1 indexing.method=fft1d")
        print("   indexing.refinement_protocol.d_min_start=1.8 indexing.stills.indexer=stills")
        print("   indexing.max_cell=45 indexing.stills.rmsd_min_px=2\"")
        sys.exit(1)

    expt_file = sys.argv[1]
    refl_file = sys.argv[2]
    
    extra_phil = None
    if len(sys.argv) > 3:
        extra_phil = sys.argv[3]
        logger.info(f"Using additional PHIL parameters string: {extra_phil}")

    run_indexing(expt_file, refl_file, extra_phil)
