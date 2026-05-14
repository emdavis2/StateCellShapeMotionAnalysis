# StateCellShapeMotionAnalysis
Code for generating figures and data in "Quantitative analysis of fibroblast migration reveals migratory states characterized by force generation, cell shape and motion" <https://doi.org/10.64898/2026.05.06.723282>

## Set up
First, to recreate the figures in the paper, the data must be downloaded from the following location: <https://drive.google.com/file/d/1ySPlUcc_0dpDjdpjCfy1bH3l7EOZfii3/view?usp=sharing>
After opening the zip file, ensure the directory (titled 'data') is located in the same directory as this repository.

To create the conda environment, run the following: 
```
conda env create -f shape_motion_analysis.yml
```

Next, activate the conda environment: `conda activate shape_motion_analysis`

## Generating figures in paper

To generate Figures 2, 5, S2, S4 (C, D), and S5, run the file `Plot_Corr_MeanMed.py`

To generate Figures 4, 7, S4 (A, B), and S9, run the file `Plot_Stripplots.py`

To generate Figures 3 (A, E, F, G, H) and S3, run the file `HMM3States_WT.py`

To generate Figures 6 (A, E, F, G, H) and S7, run the file `HMM3States_ARPC2KO.py`

To generate Figures S3 A and S7A, run the file `AICBIC_WT.py` and `AICBIC_ARPC2KO.py`

To generate Figure 3 (B, C), run the file `Plot_States_For_Track_WT_3States.py`

To generate Figure 6 (B, C), run the file `Plot_States_For_Track_ARPC2KO_3States.py`

To generate Figures 8 and S10, run the file `Plot_Protrusion_3States.py`

To generate Figure S6, run the file `HMM3_CalculateTransitionRates.py` and `HMM3_Transition_Stats.py`

To generate Figure S1, run the file `Plot_Cell_Stats.py`

## Code mentioned in Methods section
### Image processing
Once masks are generated, `ProcessMicroscopyData/gen_driftcorrection.py` is used to drift correct the movies and `ProcessMicroscopyData/gen_cellmetrics.py` is to filter out cells based on area and those that touch the edge of the frame.

### Tracking, shape, and motion metric calculation
After `ProcessMicroscopyData/gen_cellmetrics.py` is run, tracking is performed using `ProcessMicroscopyData/trackqc_savetrackmovies.py`. Shape and motion metric calculations are generated for each track with `ProcessMicroscopyData/gen_tracksgeo.py`. The output is a `pkl` file that contains the shape and motion data. Cropping the movies around each track to reduce the background was performed using `ProcessMicroscopyData/get_beadsmovies.py`.

### Gaussian hidden markov model (HMM)
The determination of the ideal number of states was found from `AICBIC_WT.py` and `AICBIC_ARPC2KO.py`. The Gaussian HMM was fit and implemented in `HMM3States_WT.py` and `HMM3States_ARPC2KO.py`.

### Protrusion measurement
The width, length, and protrusion number was found using `ProcessMicroscopyData/GetProtrusions.py`.

### Statistical analysis
The permutation test performed can be found in `stat_tests.py`.
