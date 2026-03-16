import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ldpc.simulation import simulate_ldpc
from ldpc.plotting import plot_ldpc_results

def main():
    ldpc_results, llr_snapshot = simulate_ldpc()
    plot_ldpc_results(ldpc_results, llr_snapshot)

if __name__ == "__main__":
    main()
