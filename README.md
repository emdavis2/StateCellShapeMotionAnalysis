# StateCellShapeMotionAnalysis
Code for generating figures and data in "Quantitative analysis of fibroblast migration reveals migratory states characterized by force generation, cell shape and motion" <https://doi.org/10.64898/2026.05.06.723282>

================

## Generating figures in paper

First, to recreate the figures in the paper, the data must be downloaded from the following location: <https://drive.google.com/file/d/1ySPlUcc_0dpDjdpjCfy1bH3l7EOZfii3/view?usp=sharing>
After opening the zip file, ensure the directory (titled "data") is located in the same directory as this repository.

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
