# Getting the framework to run
The framework is not intended to run out of the box, so special cases upon installation are not dealt with.
The error messages should give good guidance on what needs to be adjusted, but here are some things that definitely need adjustment:
- `Datasets.py`: the path to the dataset is saved statically and needs to be adjusted
- some folders need to created first in the working directory (which should be the directory containing the files)
  - `checkpoints`
  - `statistics`
  - `submissions`
# Structure of the framework
- `Datasets.py`: self-explanatory
- `Models.py`: self-explanatory
- `Utility.py`: various utility functions that did not fit in the previous two files
- `base[...].py`: an experiment. `base.py` is the unmodified version, all the other versions are slightly modified and saved separately for reproducability
- `submission.py`: for generating an existing an existing generator to generate images for submission
- `Analysis.ipynb`: for quick and dirty investigation of the results
