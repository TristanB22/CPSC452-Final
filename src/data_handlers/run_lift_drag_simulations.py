# standard library imports
import os
import sys

# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.simulations import XfoilSimulator

# third party imports


def run_simulations_uiuc_airfoils():
    sim = XfoilSimulator()
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    dir_list = os.listdir(data_dir)
    for subdir in dir_list:
        target_dir = os.path.join(data_dir, subdir)
        target_file = os.path.join(
            data_dir, subdir, f"{subdir}_reformatted_full_points.dat"
        )

        # try:
        sim.run_simulation(target_dir, target_file)
        # except Exception as e:
        #     print(e)
        #     continue


if __name__ == "__main__":
    run_simulations_uiuc_airfoils()
