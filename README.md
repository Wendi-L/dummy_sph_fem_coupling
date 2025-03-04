# Demo of SPH coupled with FEM through MUI

## Introduction
A 3-D flexible beam, which is 5m height, 5m deep and 2m thick, clamped at the bottom. SPH code supposed to calculate fluid forces acting on the beam, while the FEM code supposed to calculate the beam deflections. MUI is used to pass forces of SPH interface particles from SPH code to FEM code, and pass deflections of FEM interface grid points from FEM code to SPH code.

Note: internal particles/points are omitted in dummy files for simplicity.

## Usage

To run this demo you need to install FEniCSx-v0.7.2 through Spack. Once it is done, obtain a copy of the [MUI code (at least version 2.0)](https://github.com/MxUI/MUI) and install the Python wrapper of MUI under the Spack environment.

Use the following command to run the demo:

```bash
bash run_case.sh
```
A log file "output.log" will be generated.

If you have not installed MUI system wide as a CMake package and you encounter an error related to `mui.h` not being found during compilation, you can resolve this issue by specifying the path to MUI using the command:

```bash
bash run_case.sh /path/to/MUI
```
a `mpic++` wrapper with C++11 enabled backend and Python wrappers of MUI
To clean the demo directory:

```bash
bash clean_case.sh
```