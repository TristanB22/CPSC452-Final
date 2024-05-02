# standard library imports
import math
import os
import shutil

# third party imports
import numpy as np

# local imports


class AirfoilPreprocessor:
    def __init__(
        self,
        save_folder: str = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "processed"
        ),
    ):
        self.save_folder = save_folder
        self.raw_folder = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "raw"
        )

    def calculate_centroid(self, points):
        x_mean = sum(p[0] for p in points) / len(points)
        y_mean = sum(p[1] for p in points) / len(points)

        return (x_mean, y_mean)

    def angle_from_horizontal(self, centroid, point):
        dx = point[0] - centroid[0]
        dy = point[1] - centroid[1]

        return math.atan2(dy, dx)

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

    def interpolate_or_decimate_points(self, points, target_count=100):

        # convert to np
        points_array = np.array(points)

        # rightmost
        rightmost_point_index = np.argmax(points_array[:, 0])

        # split points
        upper_points = points_array[: rightmost_point_index + 1]
        lower_points = points_array[rightmost_point_index:]

        # interpolate the points with this function
        def interpolate_set(set_points, set_target_count):

            # calculate the factor for interpolation or decimation
            set_current_count = len(set_points)
            x_coords, y_coords = set_points[:, 0], set_points[:, 1]

            if set_current_count == set_target_count:
                return set_points

            else:
                # generate new linspace for interpolation
                x_new = np.linspace(x_coords[0], x_coords[-1], set_target_count)
                # y_new = np.interp(x_new, x_coords, y_coords)
                regression_coeffs = np.polyfit(x_coords, y_coords, 3)
                y_new = np.polyval(regression_coeffs, x_new)

                return np.stack((x_new, y_new), axis=-1)

        # half on top and half on bottom
        section_target_count = target_count // 2

        # add or remove top and bottom
        upper_interpolated = interpolate_set(upper_points, section_target_count)
        lower_interpolated = interpolate_set(lower_points, section_target_count)

        # combine the arrays
        return np.vstack((upper_interpolated, lower_interpolated))

    def reformat_airfoil_data(
        self,
        filename,
        new_points=None,
        entire_shuffle=False,
        interpolate=True,
    ):
        file_path = os.path.join(self.raw_folder, filename)

        # get the data points
        if new_points is None:
            points = self.get_points_from_dat_file(file_path)
        else:
            points = new_points

        # remove random points that aren't right
        points = [p for p in points if -1 <= p[0] <= 1 and -1 <= p[1] <= 1]
        points = np.array(points)

        if len(points) < 4:
            print("Not enough points to fit a spline.")
            return

        # identify most left and most right points
        min_x = min(points, key=lambda p: p[0])[0]
        max_x = max(points, key=lambda p: p[0])[0]

        # split points into upper and lower using y-value comparisons
        leftmost_point = min(
            filter(lambda p: p[0] == min_x, points), key=lambda p: p[1]
        )
        rightmost_point = min(
            filter(lambda p: p[0] == max_x, points), key=lambda p: p[1]
        )

        # normalize the entire data array to between zero and 1 x
        points = (points - min_x) / (max_x - min_x)

        if entire_shuffle:

            upper_points = [
                p
                for p in points
                if (p[1] < (leftmost_point[1] + rightmost_point[1]) / 2 and p[1] <= 1)
            ]
            lower_points = [
                p
                for p in points
                if (p[1] >= (leftmost_point[1] + rightmost_point[1]) / 2 and p[1] >= -1)
            ]

            # calculate modified centroids for sorting
            try:
                centroid_upper = self.calculate_centroid(upper_points)
                centroid_lower = self.calculate_centroid(lower_points)
            except:
                print(upper_points)
                print(lower_points)
                print((leftmost_point[1] + rightmost_point[1]))
                print((leftmost_point[1] + rightmost_point[1]))
                print(points)
                return

            # adjust centroid y-values
            centroid_upper = (centroid_upper[0], centroid_upper[1] - 10)
            centroid_lower = (centroid_lower[0], centroid_lower[1] + 10)

            # sort upper and lower points based on angle from horizontal line through centroid
            upper_points.sort(
                key=lambda p: self.angle_from_horizontal(centroid_upper, p)
            )
            lower_points.sort(
                key=lambda p: self.angle_from_horizontal(centroid_lower, p)
            )

            # re-combine the points list starting from the closest to (0,0) if exists
            points = upper_points + lower_points

        # insert zero
        try:

            # move the zero to the front
            # index = points.index((0, 0))
            zero_indexes = np.where(np.all(points == (0, 0), axis=1))

            if len(zero_indexes[0]) > 0:
                index = np.where(np.all(points == (0, 0), axis=1))[0][0]
            else:
                raise ValueError("No elements in the list")

        except ValueError:

            # insert (0,0) and find the next minimum x+y point
            np.insert(points, 0, (0, 0), axis=0)
            next_point = min(points[1:], key=lambda p: p[0] + p[1])
            points_array = np.array(points)
            index = np.where((points_array == np.array(next_point)).all(axis=1))[0][0]

        t_points = [v for v in points[index:]]
        for i in range(index):
            t_points.append(points[i])
        points = np.array(t_points)

        # check that we are rotating clockwise
        if points[1][1] < points[0][1]:
            points = points[::-1]
            points = np.concatenate(([points[-1]], points[:-1]))

        # check whether we have two sets of airfoil (top and bottom)
        # we check this by seeing how many times x changes relative sign to previous x
        # and by how much
        last_x = 0

        for curr_idx, (x, y) in enumerate(points):
            if x < last_x and last_x - x > 0.8:
                # concatenate the reversed points
                points = np.concatenate([points[:curr_idx], points[curr_idx:][::-1]])
                break
            last_x = x

        # changing the interpolation
        points = self.interpolate_or_decimate_points(points)

        try:
            # move the zero to the front
            # index = points.index((0, 0))
            zero_indexes = np.where(np.all(points == (0, 0), axis=1))

            if len(zero_indexes[0]) > 0:
                index = np.where(np.all(points == (0, 0), axis=1))[0][0]
            else:
                raise ValueError("No elements in the list")

        except ValueError:

            # insert (0,0) and find the next minimum x+y point
            next_point = min(points[1:], key=lambda p: p[0] + p[1])
            points_array = np.array(points)
            index = np.where((points_array == np.array(next_point)).all(axis=1))[0][0]

        # check if the points are reversed
        if points[index - 1][1] > points[index][1]:
            points = points[::-1]

        else:

            t_points = [v for v in points[index:]]
            for i in range(index):
                t_points.append(points[i])
            points = np.array(t_points)[1:]
            points = np.insert(points, 0, (0, 0), axis=0)

        # save the reformatted data to a new file
        new_file_path = os.path.join(
            self.save_folder,
            "reformatted",
            filename.replace(".dat", "_reformatted.dat"),
        )

        with open(new_file_path, "w") as new_file:
            for x, y in points:
                new_file.write(f"{x:.6f}\t{y:.6f}\n")

    def fix_for_xfoil(self, filename):
        file_path = os.path.join(
            self.save_folder,
            "reformatted",
            filename.replace(".dat", "_reformatted.dat"),
        )
        points = self.get_points_from_dat_file(file_path)
        points = points + [points[0]]

        new_file_path = os.path.join(
            self.save_folder,
            "reformatted_full_points",
            filename.replace("_reformatted.dat", "_reformatted_full_points.dat"),
        )

        with open(new_file_path, "w") as new_file:
            for x, y in points:
                new_file.write(f"{x:.6f}\t{y:.6f}\n")

    def __call__(self):
        dir_list = os.listdir(self.raw_folder)
        for filename in dir_list:
            if filename.endswith(".dat"):
                self.reformat_airfoil_data(filename)
                self.fix_for_xfoil(filename)

        # zip the folder for easy transport + GitHub upload
        shutil.make_archive(self.save_folder, "zip", self.save_folder)


if __name__ == "__main__":
    preprocessor = AirfoilPreprocessor()
    preprocessor()
