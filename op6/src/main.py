"""Main entry for genetic package-aware chiplet-system optimizer.

Usage example:
    python main.py <opdir.json> <gacfg.json> <sys.json> <sysmod.json>
"""
import numpy as np
import multiprocessing
import sys

from utils.logger import log_init, log_get
from utils.io_utils import load_file_json
from rules.connectivity import (
    assign_chiplet_ids,
    extract_pairwise_connections,
    parse_slack_bounds,
    parse_chiplet_dimensions,
)
from ga.pipeline import construct_chromosome_bounds
from ga.pymoo_pipeline import run_pymoo_nsga2
from debug import visualize_interactive_population

# Setup logging
log_init()
logger = log_get(__name__)


def main():
    """Program main entry."""

    # 1) Process command line inputs
    if len(sys.argv) != 5:
        logger.error(
            "Invalid argument count, use: main.py <opdir.json> <gacfg.json> <sys.json> <sysmod.json>"
        )
        raise ValueError("Invalid argument count!")

    # 2) Load configuration files
    cfg_opdir = load_file_json(sys.argv[1])
    cfg_ga = load_file_json(sys.argv[2])
    cfg_sys = load_file_json(sys.argv[3])
    cfg_sysmod = load_file_json(sys.argv[4])

    # 3) Parse system topology
    chiplet_ids = assign_chiplet_ids(cfg_sys)
    chiplet_count = len(chiplet_ids)
    chiplet_edges = extract_pairwise_connections(cfg_sys, chiplet_ids)
    chiplet_dimensions = parse_chiplet_dimensions(cfg_sys, chiplet_ids)
    chiplet_min_slack, chiplet_max_slack = parse_slack_bounds(cfg_sysmod, chiplet_ids)

    # 4) Construct Chromosome Bounds
    xl, xu = construct_chromosome_bounds(chiplet_count, cfg_ga["slack_steps"])

    # 5) Setup Multiprocessing Pool
    cpu_workers = cfg_opdir.get("cpu_workers", 1)
    pool = multiprocessing.Pool(processes=cpu_workers) if cpu_workers > 1 else None

    try:
        # 6) Run Optimization Pipeline via pymoo NSGA-II
        pop_x, pop_f, is_pareto = run_pymoo_nsga2(
            num_chiplets=chiplet_count,
            chiplet_dimensions=chiplet_dimensions,
            min_slack=chiplet_min_slack,
            max_slack=chiplet_max_slack,
            chiplet_edges=chiplet_edges,
            xl=xl,
            xu=xu,
            cfg_opdir=cfg_opdir,
            cfg_ga=cfg_ga,
            pool=pool,
        )
        logger.info("pymoo NSGA-II optimization finished successfully.")

        # 7) Launch Interactive Debug GUI Explorer
        total_samples = len(pop_x)
        pareto_samples = int(np.sum(is_pareto))
        unique_pareto_points = len(np.unique(pop_f[is_pareto], axis=0))
        unique_total_points = len(np.unique(pop_f, axis=0))

        logger.info(f"Optimization Summary:")
        logger.info(f"  Total Surviving Samples : {total_samples}")
        logger.info(f"  Pareto-Optimal Samples  : {pareto_samples} ({unique_pareto_points} unique objective pairs)")
        logger.info(f"  Suboptimal Samples      : {total_samples - pareto_samples}")
        logger.info(f"  Total Unique Objectives : {unique_total_points}")

        if pop_x is not None and len(pop_x) > 0:
            logger.info("Opening interactive population floorplan explorer window...")
            visualize_interactive_population(
                pop_x=pop_x,
                pop_f=pop_f,
                is_pareto=is_pareto,
                chiplet_id_map=chiplet_ids,
                chiplet_dimensions=chiplet_dimensions,
                min_slack=chiplet_min_slack,
                max_slack=chiplet_max_slack,
                slack_steps=cfg_ga["slack_steps"],
                chiplet_edges=chiplet_edges,
                objective_names=["Area (mm²)", "Wirelength (mm)"],
            )

    finally:
        # 8) Resource Cleanup
        if pool:
            pool.close()
            pool.join()
            logger.info("Worker pool shut down cleanly.")


if __name__ == "__main__":
    main()