import os
import matplotlib.pyplot as plt
import numpy as np
from yamvp import venn

if __name__ == "__main__":
    os.makedirs("img", exist_ok=True)
    
    # 1-set
    fig = venn([None, "A"], ["Alpha"])
    fig.savefig("img/venn1_demo.png", dpi=350, bbox_inches="tight")
    plt.close(fig)

    # 2-set
    fig = venn([[None, "B"], ["A", "AB"]], ["Alpha", "Beta"])
    fig.savefig("img/venn2_demo.png", dpi=350, bbox_inches="tight")
    plt.close(fig)

    # 3-set
    vals3 = [
        [[None, "C"], ["B", "BC"]],
        [["A", "AC"], ["AB", "ABC"]],
    ]
    fig = venn(vals3, ["Alpha", "Beta", "Gamma"])
    fig.savefig("img/venn3_demo.png", dpi=350, bbox_inches="tight")
    plt.close(fig)

    # 4-set
    vals4 = np.empty((2, 2, 2, 2), dtype=object)
    for i, yA in enumerate(("", "A")):
        for j, yB in enumerate(("", "B")):
            for k, yC in enumerate(("", "C")):
                for l, yD in enumerate(("", "D")):
                    vals4[i, j, k, l] = yA + yB + yC + yD
    vals4[0,0,0,0] = None
    #vals4[0,0,0,0] = "Other"

    fig = venn(vals4, ["Alpha", "Beta", "Gamma", "Delta"])
    fig.savefig("img/venn4_demo.png", dpi=350, bbox_inches="tight")
    plt.close(fig)

    # 5-set
    vals5 = np.empty((2, 2, 2, 2, 2), dtype=object)
    for i, yA in enumerate(("", "A")):
        for j, yB in enumerate(("", "B")):
            for k, yC in enumerate(("", "C")):
                for l, yD in enumerate(("", "D")):
                    for m, yE in enumerate(("", "E")):
                        vals5[i, j, k, l, m] = yA + yB + yC + yD + yE
    vals5[0,0,0,0,0] = None
    #vals5[0,0,0,0,0] = "Other"
    
    fig = venn(vals5, ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"])
    fig.savefig("img/venn5_demo.png", dpi=350, bbox_inches="tight")
    plt.close(fig)
