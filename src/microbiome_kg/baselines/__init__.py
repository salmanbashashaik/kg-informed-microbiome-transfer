"""Classical baseline implementations."""

from microbiome_kg.baselines.random_forest import (
    run_intersection_rf, run_zerofill_rf, run_taxonomy_rf
)
from microbiome_kg.baselines.coral import run_coral_rf, coral_transform
