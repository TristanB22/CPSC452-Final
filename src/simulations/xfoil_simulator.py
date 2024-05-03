"""
code for generating airfoil designs using online airfoil datasets
code is inspired by:
The Department of Energy [https://catalog.data.gov/dataset/airfoil-computational-fluid-dynamics-2k-shapes-25-aoas-3-re-numbers]

x/c: The x-coordinate normalized by the chord length.
y/c: The y-coordinate normalized by the chord length.
Ue/Vinf: Edge velocity divided by the free stream velocity.
Dstar: Displacement thickness.
Theta: Momentum thickness.
Cf: Skin friction coefficient.
H: Shape factor.
"""

# standard library imports
import os
import pickle
import platform
import subprocess

# third party imports
import matplotlib.pyplot as plt
import pandas as pd

# local imports


class XfoilSimulator:
    MAC_PLATFORM = 1
    LINUX_PLATFORM = 2
    WINDOWS_PLATFORM = 3
    UNKNOWN_PLATFORM = 4
    PLATFORM = UNKNOWN_PLATFORM

    def __init__(
        self,
        verbose,
        reynolds_number=1e6,
        mach_number=0.3,
        s_aot=2.0,
        t_aot=6.0,
        aot_increments=1.0,
    ):
        self.VERBOSE = verbose
        self.reynolds_number = reynolds_number
        self.mach_number = mach_number
        self.s_aot = s_aot
        self.t_aot = t_aot
        self.aot_increments = aot_increments

        # Set OS platform variable
        os_type = platform.system()
        if os_type == "Darwin":
            self.PLATFORM = self.MAC_PLATFORM
        elif os_type == "Linux":
            self.PLATFORM = self.LINUX_PLATFORM
        elif os_type == "Windows":
            self.PLATFORM = self.WINDOWS_PLATFORM
        else:
            self.PLATFORM = self.UNKNOWN_PLATFORM

        # Set XFOIL command based on platform
        if self.PLATFORM == self.MAC_PLATFORM:
            self.XFOIL_COMMAND = os.path.join(
                os.path.dirname(__file__), "Xfoil-for-Mac", "bin", "xfoil"
            )
        elif self.PLATFORM == self.LINUX_PLATFORM:
            # this assumes xfoil is in the PATH and was installed using
            # `sudo apt-get install xfoil`
            self.XFOIL_COMMAND = "xfoil"
        elif self.PLATFORM == self.WINDOWS_PLATFORM:
            self.XFOIL_COMMAND = os.path.join(
                os.path.dirname(__file__), "XFOIL6.99", "xfoil.exe"
            )
        else:
            raise NotImplementedError("Running program on unsupported platform.")

    def pass_to_xfoil(self, command_array):

        if self.VERBOSE:
            print(self.XFOIL_COMMAND)
            print(command_array)

        process = subprocess.Popen(
            [self.XFOIL_COMMAND],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # get the output and errors
        output, _ = process.communicate(command_array)

        return output

    def cut_string_at_keyword(self, text, keyword):

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

    def get_airfoil_information(self, xfoil_output):
        # get the thickness and the camber of the airfoil
        print(xfoil_output)
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
                max_thickness_string.index("=")
                + 1 : max_thickness_string.index("at x = ")
            ].strip()
        )

        # get the thickness and the camber of the airfoil
        max_camber_index = xfoil_output.index("Max camber") + len("Max camber")
        max_camber_end_of_line = xfoil_output.find("\n", max_camber_index)
        max_camber_string = xfoil_output[
            max_camber_index:max_camber_end_of_line
        ].strip()

        # get the value and the x pos
        max_camber_x = float(
            max_camber_string[
                max_camber_string.index("at x = ") + len("at x = ") :
            ].strip()
        )
        max_camber_val = float(
            max_camber_string[
                max_camber_string.index("=") + 1 : max_camber_string.index("at x = ")
            ].strip()
        )

        # get everything up to "j/t"
        xfoil_output = self.cut_string_at_keyword(xfoil_output.lower(), "j/t")

        # parsing for the start of the bend info
        if self.VERBOSE:
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
        print(xfoil_bend_metrics)
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

                if self.VERBOSE:
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

    def kill_xquartz(self):
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

    def get_points_from_dat_file(self, file_path):

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

    def show_airfoil(self, point_array=None, file_path=None):

        # check that the arguments are valid
        if point_array is None and file_path is None:
            raise ValueError("Improper arguments passed to the file")

        # if the file path is where we should be pulling from then get it
        if file_path is not None:
            point_array = self.get_points_from_dat_file(file_path)

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

    def run_simulation(self, target_dir, target_file):
        output_file = "polar.txt"
        dump_file = "dump.txt"
        target_polar = os.path.join(target_dir, output_file)
        target_dump = os.path.join(target_dir, dump_file)

        # fixing the data file

        # run the simulation on the target directory
        airfoil_commands = f"""
        LOAD {target_file}
        OPER        
        OPER             
        PACC
        {target_polar}
        {target_dump}
        M {self.mach_number}
        VISC {self.reynolds_number}
        ASEQ {self.s_aot} {self.t_aot} {self.aot_increments}
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
        program_output = self.pass_to_xfoil(airfoil_commands)

        if self.VERBOSE:
            print(program_output)

        airfoil_commands = f"""
        LOAD {target_file.replace("_reformatted.dat", "_reformatted_full_points.dat")}   
        t_foil   
        BEND
        QUIT
        """

        # getting the dictionary information
        program_output = self.pass_to_xfoil(airfoil_commands)

        if self.VERBOSE:
            print(airfoil_commands)
            print(program_output)

        # kill the XQuartz
        if self.PLATFORM == self.MAC_PLATFORM:
            self.kill_xquartz()

        # getting the airfoil information
        airfoil_info_dict = self.get_airfoil_information(program_output)

        with open(os.path.join(target_dir, "airfoil_info.pkl"), "wb") as f:
            pickle.dump(airfoil_info_dict, f)

        if self.VERBOSE:
            print(airfoil_info_dict)

        # read the resulting csv for the information that we are looking for
        resulting_data = pd.read_csv(target_polar, sep="\s+", skiprows=10)  # type: ignore
        resulting_data = resulting_data.drop(0)

        if self.VERBOSE:
            print(resulting_data)
