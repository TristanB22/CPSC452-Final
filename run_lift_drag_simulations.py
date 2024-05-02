# code for generating airfoil designs using online airfoil datasets
# code is inspired by:
# The Department of Energy [https://catalog.data.gov/dataset/airfoil-computational-fluid-dynamics-2k-shapes-25-aoas-3-re-numbers]

import math
import os
import pickle
import platform
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import sklearn
import tqdm

# state variables
MAC_PLATFORM = 1
LINUX_PLATFORM = 2
UNKNOWN_PLATFORM = 3
PLATFORM = UNKNOWN_PLATFORM

# get the platform
os_type = platform.system()
if os_type == "Darwin":
    PLATFORM = MAC_PLATFORM
elif os_type == "Linux":
    PLATFORM = LINUX_PLATFORM
else:
    PLATFORM = UNKNOWN_PLATFORM

# should we be verbose
VERBOSE = False

# getting the file that we are currently working in
# CURR_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
CURR_FILE_DIR = "."

# define the directory that we are working in for the example files
EXAMPLE_DIR = "Example Data"

# defining the data files that we are gonna use
output_file = "polar.txt"
example_data_output = os.path.join(CURR_FILE_DIR, output_file)
dump_file = "dump.txt"
example_data_dump = os.path.join(CURR_FILE_DIR, dump_file)


### XFOIL Params
# the reynolds number is used to determine the turbulence of the flow that we are computing
# the metrics of the wing for
# anything > 3500 is turbulent and > 4000 is fully turbulent
reynolds_number = 1e6
# reynolds_number = 0

# the mach number (compression) that we should be computing for here
mach_number = 0.3

# this is the angle of attack that we are going to compute the metrics for
# the starting angle of attack
s_aot = 2.0

# the ending angle of attack
t_aot = 6.0

# increments for the angle of attack
aot_increments = 1.0


# define the xfoil command depending on the platform
## CHANGE THIS IF NEEDED
if PLATFORM == MAC_PLATFORM:
    home_directory = os.path.expanduser("~")
    XFOIL_COMMAND = f"{home_directory}/Desktop/Xfoil-for-Mac/bin/xfoil"
elif PLATFORM == LINUX_PLATFORM:
    XFOIL_COMMAND = "xfoil"
else:
    raise NotImplementedError("Running program on unsupported platform.")


# define a function that we can pass command arrays to for xfoil
# this is for if we only have the command line version of XFoil running
def pass_to_xfoil(command_array):

    if VERBOSE:
        print(XFOIL_COMMAND)
        print(command_array)

    # start a subprocess
    # process = subprocess.run([XFOIL_COMMAND], input=command_array, text=True, capture_output=True)

    process = subprocess.Popen(
        [XFOIL_COMMAND],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # get the output and errors
    output, errors = process.communicate(command_array)

    # return the std out
    # return process.stdout
    return output


# # define a function to interact with the XFoil package itself
# def xfoil_package_run_command(xfoil_input_commands):

#     raise NotImplementedError(
#         "Cannot import XFoil into Linux operating environment -- try another"
#     )


# # define a function that splits the commands that we pass to it to a format that xfoil can understand
# # all of the commands in XFoil are in a format where there is alphanumeric and then numbers
# def process_input_commands(input_command):

#     raise NotImplementedError(
#         "Have not implemented a function for processing the XFoil input commands yet."
#     )


# get everything before and including a line with a key word in a python string
def cut_string_at_keyword(text, keyword):

    # split the string into lines
    lines = text.split("\n")

    # go through each of the lines
    for i, line in enumerate(lines):

        # check if they keyword is in the line or not
        if keyword in line:

            # return the text up to and including the line with the keyword
            return "\n".join(lines[: i + 1])

    # if the keyword isn't found, return the original text
    return text


# get the important information about the airfoil from the following function
def get_airfoil_information(xfoil_output):

    # try:

    # get the thickness and the camber of the airfoil
    max_thickness_index = xfoil_output.index("Max thickness") + len("Max thickness")
    max_thickness_end_of_line = xfoil_output.find("\n", max_thickness_index)
    max_thickness_string = xfoil_output[
        max_thickness_index:max_thickness_end_of_line
    ].strip()

    # get the value and the x pos
    max_thickness_x = float(
        max_thickness_string[
            max_thickness_string.index("at x = ") + len("at x = ") :
        ].strip()
    )
    max_thickness_val = float(
        max_thickness_string[
            max_thickness_string.index("=") + 1 : max_thickness_string.index("at x = ")
        ].strip()
    )

    # get the thickness and the camber of the airfoil
    max_camber_index = xfoil_output.index("Max camber") + len("Max camber")
    max_camber_end_of_line = xfoil_output.find("\n", max_camber_index)
    max_camber_string = xfoil_output[max_camber_index:max_camber_end_of_line].strip()

    # get the value and the x pos
    max_camber_x = float(
        max_camber_string[max_camber_string.index("at x = ") + len("at x = ") :].strip()
    )
    max_camber_val = float(
        max_camber_string[
            max_camber_string.index("=") + 1 : max_camber_string.index("at x = ")
        ].strip()
    )

    # get everything up to "j/t"
    xfoil_output = cut_string_at_keyword(xfoil_output.lower(), "j/t")

    # parsing for the start of the bend info
    if VERBOSE:
        print(xfoil_output)
    bend_info_start = xfoil_output.index("area =")

    # cutting the string
    xfoil_bend_metrics = xfoil_output[bend_info_start:]

    # go through and get all of the metrics from xfoil and return it to a dictionary
    # initialize an empty dictionary
    properties_dict = {}

    current_category = None

    # split the text by newlines and iterate through the lines
    lines = xfoil_bend_metrics.split("\n")
    for line in lines:

        # check the value and the key
        if "=" in line:

            # get the key and value
            key, value = line.split("=")
            key = key.strip()
            value = value.strip()

            if current_category is None or key == "j/t" or key == "j":
                properties_dict[key] = (
                    float(value) if "e" in value or "." in value else int(value)
                )
            else:
                properties_dict[current_category][key] = (
                    float(value) if "e" in value or "." in value else int(value)
                )

            if VERBOSE:
                print(
                    f"current_category: {current_category}\nKey: {key}\nValue: {value}\n"
                )

        elif "parameters" in line:
            category, material = line.split("(")
            category = category.strip().replace("-bending ", "-")
            material = material.strip("):")
            current_category = category + "_" + material
            properties_dict[current_category] = {}

    # adding the other information that we got at the start to the dictionary
    properties_dict["max_thickness_x"] = max_thickness_x
    properties_dict["max_thickness_val"] = max_thickness_val
    properties_dict["max_camber_x"] = max_camber_x
    properties_dict["max_camber_val"] = max_camber_val

    return properties_dict


def kill_xquartz():
    try:
        # find the pids
        ascript = """
        tell application "System Events"
            set xquartzProcesses to every process whose name is "quartz-wm"
            repeat with proc in xquartzProcesses
                try
                    do shell script ("kill -9 " & (unix id of proc))
                end try
            end repeat
        end tell
        """

        # kill each process found
        subprocess.run(["osascript", "-e", ascript])
        print(f"Killed XQuartz process")

    except subprocess.CalledProcessError:
        print("XQuartz is not running.")


# define a function that gets the airfoil information for attack angles
# def get_lift_drag_information():

#     return drag_lift_dict


# getting the points from a file path
def get_points_from_dat_file(file_path):

    # open the file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # filter out non-numeric lines and strip whitespace
    points = []
    for line in lines:

        parts = line.strip().split()
        if len(parts) == 2:
            try:
                points.append((float(parts[0]), float(parts[1])))
            except ValueError:
                print(f"Skipping invalid line: {line}")
        else:
            print(parts)

    return points


# generate coordinates for a NACA 4210 airfoil.
def show_airfoil(point_array=None, file_path=None):

    # check that the arguments are valid
    if point_array is None and file_path is None:
        raise ValueError("Improper arguments passed to the file")

    # if the file path is where we should be pulling from then get it
    if file_path is not None:
        point_array = get_points_from_dat_file(file_path)

    assert point_array is not None

    # split the points properly given that we have tuples
    xu, yu, xl, yl = [], [], [], []
    for i, point in enumerate(point_array):
        if i < len(point_array) / 2:
            xu.append(point[0])
            yu.append(point[1])
        else:
            xl.append(point[0])
            yl.append(point[1])

    for i in range(len(xu) > len(xl)):
        xu.append(xu[-1])
        yu.append(yu[-1])

    for i in range(len(xu) - len(xl)):
        xu.append(xu[-1])
        yu.append(yu[-1])

    xu.reverse()

    xl.append(xu[-1])
    yl.append(yu[-1])
    xu.append(xl[-1])
    yu.append(yl[-1])

    # plot airfoil
    plt.figure(figsize=(10, 5))
    plt.plot(xu, yu, "b", xl, yl, "r")  # upper in blue, lower in red
    plt.fill_between(xu, yu, yl, color="gray", alpha=0.5)
    plt.axis("equal")
    plt.title("NACA 4210 Airfoil" if point_array is None else "Custom Airfoil")
    plt.xlabel("Chord")
    plt.ylabel("Thickness")
    plt.grid(True)
    plt.show()


def run_simulation(target_dir, target_file):

    target_polar = os.path.join(target_dir, output_file)
    target_dump = os.path.join(target_dir, dump_file)

    # fixing the data file

    # run the simulation on the target directory
    airfoil_commands = f"""
    LOAD {target_file.replace("_reformatted.dat", "_reformatted_full_points.dat")}
    OPER        
    OPER             
    PACC
    {target_polar}
    {target_dump}
    M {mach_number}
    VISC {reynolds_number}
    ASEQ {s_aot} {t_aot} {aot_increments}
    DUMP {dump_file}
    PACC

    QUIT
    """

    # remove the files so that the program runs smoothly
    try:
        os.remove(target_polar)
    except:
        pass

    try:
        os.remove(target_dump)
    except:
        pass

    # run xfoil to get the airfoil metrics
    program_output = pass_to_xfoil(airfoil_commands)

    if VERBOSE:
        print(program_output)

    airfoil_commands = f"""
    LOAD {target_file.replace("_reformatted.dat", "_reformatted_full_points.dat")}   
    t_foil   
    BEND
    QUIT
    """

    # getting the dictionary information
    program_output = pass_to_xfoil(airfoil_commands)

    if VERBOSE:
        print(airfoil_commands)
        print(program_output)

    # kill the XQuartz
    if PLATFORM == MAC_PLATFORM:
        kill_xquartz()

    # getting the airfoil information
    airfoil_info_dict = get_airfoil_information(program_output)

    with open(os.path.join(target_dir, "airfoil_info.pkl"), "wb") as f:
        pickle.dump(airfoil_info_dict, f)

    if VERBOSE:
        print(airfoil_info_dict)

    # read the resulting csv for the information that we are looking for
    resulting_data = pd.read_csv(target_polar, sep="\s+", skiprows=10)
    resulting_data = resulting_data.drop(0)

    if VERBOSE:
        print(resulting_data)


# run the simulation for each of the dat files
if __name__ == "__main__":

    # each of the potential directories
    # dir_list = os.listdir("./Example Data")
    dir_list = os.listdir(".")

    # check each of the potential files
    for potential_file in tqdm.tqdm(dir_list):

        # run the simulation
        # pot_dir = os.path.join("./Example Data", potential_file)
        pot_dir = os.path.join(".", potential_file)

        # print(f"Simulating {pot_dir}")

        try:

            run_simulation(
                pot_dir, os.path.join(pot_dir, f"{potential_file}_reformatted.dat")
            )
            # show_airfoil(file_path = os.path.join(pot_dir, f"{potential_file}.dat"))
            # show_airfoil(file_path = os.path.join(pot_dir, f"{potential_file}_reformatted.dat"))
        except Exception as e:
            print(e)
            pass


# write out example commands that we can send to the XFoil program


# VISC {reynolds_number}
# LOAD {example_data_path}

# this is the output to the dump file (according to ChatGPT):
"""
x/c: The x-coordinate normalized by the chord length.
y/c: The y-coordinate normalized by the chord length.
Ue/Vinf: Edge velocity divided by the free stream velocity.
Dstar: Displacement thickness.
Theta: Momentum thickness.
Cf: Skin friction coefficient.
H: Shape factor.
"""
