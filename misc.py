""" Utility functions for formatting/transforming data from output."""


def print_metrics(x):
    """Format and print to add to table in latex."""
    print("Actual acc: ", round(x["actual_acc"], 2))
    print("Actual kappa: ", round(x["actual_kappa"], 2))
    for m in ["random", "uncertainty", "reward"]:
        print(m)
        y = [
            x["%s_acc" % m][4],
            x["%s_acc" % m][9],
            x["%s_acc" % m][19],
            x["%s_acc" % m][29],
            x["%s_acc" % m][39],
            x["%s_kappa" % m][4],
            x["%s_kappa" % m][9],
            x["%s_kappa" % m][19],
            x["%s_kappa" % m][29],
            x["%s_kappa" % m][39],
        ]
        y = [str(round(i, 2)) for i in y]
        print(" & ".join(y))
        print("---")


def print_30_metrics(x):
    for m in ["random", "uncertainty", "reward"]:
        print(m, round(x["%s_acc" % m][14], 2), round(x["%s_kappa" % m][14], 2))
