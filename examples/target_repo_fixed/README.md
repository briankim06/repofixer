# Target Repo (Clean Baseline)

This directory contains a **clean, immutable baseline** of the tiny calculator
package used for generating synthetic training data.

All unit tests pass (`pytest -q` yields 0 failures). Synthetic data generation
makes **copies** of this directory, mutates them, and never modifies this
baseline in-place.