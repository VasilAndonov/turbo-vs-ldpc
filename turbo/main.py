import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turbo.simulation import run_convolutional_baseline, simulate_turbo
from turbo.plotting import plot_turbo_results

def main():
    uncoded, coded = run_convolutional_baseline()
    turbo_results, llr_snapshot = simulate_turbo()
    plot_turbo_results(uncoded, coded, turbo_results, llr_snapshot)

if __name__ == "__main__":
    main()
