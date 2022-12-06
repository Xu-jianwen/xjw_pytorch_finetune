import os
import numpy as np
from util import cm_plot, percentage_cm_plot

cm = np.loadtxt("baseline_cm.csv", delimiter=",")
acc = cm.trace()/cm.sum()
print(acc,"  ", cm.sum())
percentage_cm_plot(
        num_classes=5,
        # label=classes,
        label=["cargo", "other", "carrier", "fishing", "tanker"],
        matrix=cm,
        fig_name="confusion_matrixs/FUSAR_" + "UMP+D_2_",
    )