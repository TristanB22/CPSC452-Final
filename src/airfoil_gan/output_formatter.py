# standard library imports

# third party imports
import bezier
import matplotlib.pyplot as plt
import numpy as np

# local imports


class OutputFormatter:
    def __init__(self, verbose, window_val, bezier_smooth):
        self.verbose = verbose
        self.window_val = window_val
        self.bezier_smooth = bezier_smooth

    def smooth_foil(self, array_y, window=5):
        # start by extending the data to make sure that the averaging function works
        array_y = np.pad(array_y, (window // 2), mode="edge")

        # compute the moving average using a kernel method
        mv_mat = np.ones(window) / window

        # resulting array
        res = np.convolve(array_y, mv_mat, mode="valid")

        return res

    def smooth_bezier(self, array_x, array_y):
        # define the nodes
        bezier_nodes = np.asfortranarray([array_x, array_y])

        # get the curve
        b_curve = bezier.Curve(bezier_nodes, degree=len(array_x) - 1)

        # generate a linear space
        b_line = np.linspace(0, 1, 100)
        res = b_curve.evaluate_multi(b_line)

        # return the x and y
        return (res[0], res[1])

    def sort_points_airfoil_edge(self, array_x, array_y, bottom=False):

        array_x = np.array(array_x)
        array_y = np.array(array_y)

        # sort the arrays using either reverse or forward and the x array
        # reverse when we are dealing with the bottom array
        sorted_indexes = np.argsort(array_x)

        # check if we should reverse
        if bottom:
            sorted_indexes = sorted_indexes[::-1]

        # sort the arrays
        array_x = array_x[sorted_indexes]
        array_y = array_y[sorted_indexes]

        return (array_x, array_y)

    def smooth_airfoil(self, array_x, array_y):
        # now get the average of the points
        # to make the airfoil more smooth
        if not self.bezier_smooth:
            array_y = self.smooth_foil(array_y, window=self.window_val)
        else:
            (array_x, array_y) = self.smooth_bezier(array_x, array_y)

        return (array_x, array_y)

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
                    # print(f"Skipping invalid line: {line}")
                    pass

        return points

    def show_airfoil(
        self, point_array=None, file_path=None, normalize=True, format_for_xfoil=True
    ):

        # check that the arguments are valid
        if point_array is None and file_path is None:
            raise ValueError("Improper arguments passed to the file")

        # if the file path is where we should be pulling from then get it
        if file_path is not None:
            point_array = self.get_points_from_dat_file(file_path)

        # if we did not read this from a file, then we are not running on tuples
        # and need to convert the points to tuples
        else:
            assert point_array is not None
            # define a new point array
            new_point_arr = []

            print(f"ADDING: {len(point_array) // 2}")

            # get all points
            for i in range(len(point_array) // 2):

                # add new tuples
                new_point_arr.append((point_array[2 * i], point_array[2 * i + 1]))

            # overwrite the old file
            point_array = new_point_arr

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
        yu.reverse()

        # now fix the top and bottom
        if normalize:
            (xu, yu) = self.sort_points_airfoil_edge(xu, yu, bottom=False)
            (xl, yl) = self.sort_points_airfoil_edge(xl, yl, bottom=False)

        # add the top and bottom of the airfoil
        xl = np.append(xl, xu[-1])
        yl = np.append(yl, yu[-1])
        xl = np.append(xu[0], xl)
        yl = np.append(yu[0], yl)

        xu = np.append(xu, xl[-1])
        yu = np.append(yu, yl[-1])
        xu = np.append(xl[0], xu)
        yu = np.append(yl[0], yu)

        # smooth the airfoil out
        (xl, yl) = self.smooth_airfoil(xl, yl)
        (xu, yu) = self.smooth_airfoil(xu, yu)

        # re-split the points so that there is a top and a bottom
        point_array = []
        for _x, _y in zip(xu, yu):
            point_array.append((_x, _y))
        for _x, _y in zip(reversed(xl), reversed(yl)):
            point_array.append((_x, _y))

        # turn into numpy
        point_array = np.array(point_array)

        # remove duplicates
        new_arr = []
        seen_pairs = set()

        # iterate through the pairs
        for pair in point_array:

            if str(pair) not in seen_pairs:

                new_arr.append(pair)

            # add the pair to the seen set
            seen_pairs.add(str(pair))

        point_array = np.array(new_arr)

        # identify most left and most right points
        # getting the min and the max x
        min_x_idx = None
        max_x_idx = None
        min_x = np.inf
        max_x = -np.inf

        for idx, pair in enumerate(point_array):
            if pair[0] < min_x:
                min_x_idx = idx
                min_x = pair[0]
            if pair[0] > max_x:
                max_x_idx = idx
                max_x = pair[0]

        # print(point_array)

        # split points into upper and lower using y-value comparisons
        leftmost_point = point_array[min_x_idx]
        rightmost_point = point_array[max_x_idx]

        # normalize the entire data array to between zero and 1
        point_array = (point_array - min_x) / (max_x - min_x)

        # getting the x and y values for top and bottom
        xu = point_array[:max_x_idx, 0]
        yu = point_array[:max_x_idx, 1]
        xl = point_array[max_x_idx:, 0]
        yl = point_array[max_x_idx:, 1]

        # go ahead and normalize it to (0, 1) on either end
        (xl, yl) = self.smooth_airfoil(xl, yl)
        (xu, yu) = self.smooth_airfoil(xu, yu)

        # change to arrays
        xl = np.array(xl)
        yl = np.array(yl)
        # yl = np.array(list(reversed(yl)))
        xu = np.array(xu)
        yu = np.array(yu)

        # normalize the x distance
        xl = (xl - min(xl)) / (max(xl) - min(xl))
        xu = (xu - min(xu)) / (max(xu) - min(xu))

        # get the initial height and final height
        back_y = yu[-1]
        front_y = yu[0]

        # normalize by different ratio across airfoil
        len_arr = len(xl)
        for i in range(len_arr):
            yu[i] -= back_y * (i / (len_arr - 1)) + front_y * (
                ((len_arr - 1) - i) / (len_arr - 1)
            )
            yl[i] -= front_y * (i / (len_arr - 1)) + back_y * (
                ((len_arr - 1) - i) / (len_arr - 1)
            )

        delta_back = yl[0]
        delta_front = yl[-1]

        for i in range(len_arr):
            yl[i] -= delta_front * (i / (len_arr - 1)) + delta_back * (
                ((len_arr - 1) - i) / (len_arr - 1)
            )

        # check if this is verbose
        if self.verbose:
            print(f"point_array len: {len(point_array)}")
            print(f"max_x_idx: {max_x_idx}")
            print(f"min_x_idx: {min_x_idx}")
            print(f"delta_back: {delta_back}")
            print(f"delta_front: {delta_front}")
            print(f"back_y: {back_y}")
            print(f"front_y: {front_y}")
            print(f"xl: {len(xl)}")
            print(f"yl: {len(yl)}")
            print(f"xu: {len(xu)}")
            print(f"yu: {len(yu)}")

        # plot airfoil
        plt.figure(figsize=(10, 5))
        plt.plot(xu, yu, "b", xl, yl, "r")  # upper in blue, lower in red
        plt.fill_between(xu, yu, list(reversed(yl)), color="gray", alpha=0.5)
        plt.axis("equal")
        plt.title("NACA 4210 Airfoil" if point_array is None else "Custom Airfoil")
        plt.xlabel("Chord")
        plt.ylabel("Thickness")
        plt.grid(True)
        plt.show()

        # get the new point array
        point_array = []
        for _x, _y in zip(xl, yl):
            point_array.append((_x, _y))
        for _x, _y in zip(xu, yu):
            point_array.append((_x, _y))

        # remove duplicates
        new_arr = []
        seen_pairs = set()

        # iterate through the pairs
        for pair in point_array:

            if str(pair) not in seen_pairs:

                new_arr.append(pair)

            # add the pair to the seen set
            seen_pairs.add(str(pair))

        point_array = np.array(new_arr)
        point_array = np.append(point_array, [[1.00, 0.00]], axis=0)

        # return the point array
        return point_array
