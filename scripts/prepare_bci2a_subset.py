#!/usr/bin/env python3
"""Convert official BCI IV-2a training GDF files into project `.npy` arrays."""

from eegclassify.cli import prepare_bci2a_main


if __name__ == "__main__":
    prepare_bci2a_main()
