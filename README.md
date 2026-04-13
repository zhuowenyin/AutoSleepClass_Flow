# Automatic Sleep Classification Workflow

This is the automatic sleep classification script used in the the paper:

Hao, L.\*, Woolley, J.\*, Yin, Z.\*, Jin, X., Stucynski, J., Ortega, R. A. S., Corder, G., Chung, S., & Weber, F. (2026). Heart rate and sleep history encode ultradian REM sleep timing. [*Current Biology*](https://www.cell.com/current-biology/fulltext/S0960-9822(26)00167-3).

The classification workflow code is written by Leilei Hao and Zhuowen Yin.

## Setup and Running Code

You can open sleep_classification_workflow.ipynb, set up prerequisite and data directories, and run the workflow in the notebook. The code relies on the PySleep modules in https://github.com/tortugar/Lab/tree/master/PySleep.

## Functionalities

The code can automatically annotate sleep stages from mouse EEG/EMG data. It has pretrained 3-sleep-stage (REM, Wake, NREM) models, supports finetuning with your own annotations, and supports adding extra sleep stages during finetuning.
