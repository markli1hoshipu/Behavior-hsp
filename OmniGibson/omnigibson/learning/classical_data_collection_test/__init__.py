"""
classical_data_collection_test - Ground-Truth Based Data Generation

This package provides tools for generating training data using ground-truth
task execution instead of ML policies. It uses simulator knowledge (known maps,
object poses, task goals) to execute tasks and generate realistic physics-based
robot trajectories.

Main components:
- GroundTruthTaskExecutor: Replaces ML policies for task execution
- BDDLGoalParser: Converts BDDL goals into action primitive sequences
- eval_data_gen_gt.py: Main script for data generation
"""

from omnigibson.learning.classical_data_collection_test.gt_task_executor import (
    GroundTruthTaskExecutor,
    BDDLGoalParser,
    PrimitiveSpec,
)

__all__ = [
    "GroundTruthTaskExecutor",
    "BDDLGoalParser",
    "PrimitiveSpec",
]
