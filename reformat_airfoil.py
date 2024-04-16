# this script takes the path to dat file as the first command line argument
# and reformats the file so that it works with xfoil
# currently, it does not interpolate the points to work with the model

import os
import sys
import math
import shutil

import numpy as np
from scipy.interpolate import splprep, splev

import numpy as np

# calculate the centroid (geometric mean) of a set of points
def calculate_centroid(points):
    x_mean = sum(p[0] for p in points) / len(points)
    y_mean = sum(p[1] for p in points) / len(points)
    return (x_mean, y_mean)

# calculate the angle from the horizontal for a point given a centroid
def angle_from_horizontal(centroid, point):
    dx = point[0] - centroid[0]
    dy = point[1] - centroid[1]
    return math.atan2(dy, dx)

# getting the points from a file path
def get_points_from_dat_file(file_path):

	# open the file
	with open(file_path, 'r') as file:
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

# change it so that it is a cycle instead
def reformat_airfoil_data(file_path, new_points = None, change_file=True, entire_shuffle = False, interpolate = True):
    
	# get the data points
	if new_points is None:
		points = get_points_from_dat_file(file_path)
	else:
		points = new_points

	# if not points:
	# 	print("no valid data points found.")
	# 	return

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
	leftmost_point = min(filter(lambda p: p[0] == min_x, points), key=lambda p: p[1])
	rightmost_point = min(filter(lambda p: p[0] == max_x, points), key=lambda p: p[1])

	# normalize the entire data array to between zero and 1 x
	points = (points - min_x) / (max_x - min_x)

	if entire_shuffle:	


		upper_points = [p for p in points if (p[1] < (leftmost_point[1] + rightmost_point[1]) / 2 and p[1] <= 1)]
		lower_points = [p for p in points if (p[1] >= (leftmost_point[1] + rightmost_point[1]) / 2 and p[1] >= -1)]

		# calculate modified centroids for sorting
		try:
			centroid_upper = calculate_centroid(upper_points)
			centroid_lower = calculate_centroid(lower_points)
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
		upper_points.sort(key=lambda p: angle_from_horizontal(centroid_upper, p))
		lower_points.sort(key=lambda p: angle_from_horizontal(centroid_lower, p))

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

	# print(f"points: {index}")
	# print([points[index:], points[:index]])
	# print()

	# concatenate the lists together
	# np.concatenate((points[index::-1], points[:index]))
 
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
	points = interpolate_or_decimate_points(points)

	# if "hs1404" in file_path:
	# 	print(points)
	# 	print()

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

	# if "hs1404" in file_path:
	# 	print(points)
	# 	exit()


	# save the reformatted data to a new file
	if change_file:
		new_file_path = file_path.replace('.dat', '_reformatted.dat')
	else:
		new_file_path = file_path

	with open(new_file_path, 'w') as new_file:
		for x, y in points:
			new_file.write(f"{x:.6f}\t{y:.6f}\n")

	# print(f"Reformatted data saved to {new_file_path}")



# def reformat_airfoil_data(file_path, new_points = None, change_file=True):
    
# 	# get the data points
# 	if new_points is None:
# 		points = get_points_from_dat_file(file_path)
# 	else:
# 		points = new_points

# 	if not points:
# 		print("no valid data points found.")
# 		return

# 	# remove random points that aren't right
# 	points = [p for p in points if -1 <= p[0] <= 1 and -1 <= p[1] <= 1]
# 	points = np.array(points)

# 	if len(points) < 4:
# 		print("Not enough points to fit a spline.")
# 		return

# 	print()
# 	print([points[:,0], points[:,1]])
# 	print()

# 	tck, u = splprep([points[:,0], points[:,1]], s=0)

# 	# evaluate the spline at many points to determine a smooth curve
# 	unew = np.linspace(0, 1, len(points))
# 	new_points = splev(unew, tck)

# 	# sort points based on the spline parameter
# 	sorted_indices = np.argsort(u)
# 	sorted_points = points[sorted_indices]

# 	# cycle the points to start with the closest to (0,0)
# 	try:
# 		# find the point closest to (0,0)
# 		index = np.argmin(np.sum(np.square(sorted_points - np.array([0, 0])), axis=1))
# 	except ValueError:
# 		print("failed to find the starting point near (0, 0)")

# 	# re-order points starting from the index found
# 	sorted_points = np.roll(sorted_points, -index, axis=0)

# 	# save the reformatted data to a new file
# 	if change_file:
# 		new_file_path = file_path.replace('.dat', '_reformatted.dat')
# 	else:
# 		new_file_path = file_path

# 	with open(new_file_path, 'w') as new_file:
# 		for x, y in sorted_points:
# 			new_file.write(f"{x:.6f}\t{y:.6f}\n")

    # print(f"reformatted data saved to {new_file_path}")




# define a function that takes a file as an input, creates a folder as an output
# and moves everything while reformatting
def create_and_move(working_directory = "Example Data"):

	# create new directory
	if not os.path.exists(working_directory):
		os.makedirs(working_directory)

	# get the files
	move_files = os.listdir(working_directory)
	move_files = [mv_f for mv_f in move_files if ".dat" in mv_f]

	# go through each of the potential files
	for file_name in move_files:

		if file_name.endswith('.dat'):
			
			# create corresponding directory
			directory_name = os.path.join(working_directory, file_name.replace('.dat', ''))
			file_path = os.path.join(working_directory, file_name)

			# create a new directory for the file
			if not os.path.exists(directory_name):
				os.makedirs(directory_name)

			# move the .dat file into the new directory
			new_file_location = os.path.join(directory_name, file_name)
			shutil.move(file_path, new_file_location)

			try:

				# run the reformat function on the new file location
				reformat_airfoil_data(new_file_location)

			except:
				print(f"Failed on {new_file_location}")

		else:
			print(f"Skipped {file_name}, not a .dat file")



# this function goes through the data files for the points and interpolates
# or removes the points so that we end up with 100 points that we can put into a machine learning model
# which is a GAN for this case
def interpolate_or_decimate_points(points, target_count=100):
    
	# convert to np
	points_array = np.array(points)

	# rightmost
	rightmost_point_index = np.argmax(points_array[:, 0])

	# split points
	upper_points = points_array[:rightmost_point_index + 1]
	lower_points = points_array[rightmost_point_index:]

	# interpolate the points with this function
	def interpolate_set(set_points, set_target_count):
		
		# calculate the factor for interpolation or decimation
		set_current_count = len(set_points)
		x_coords, y_coords = set_points[:, 0], set_points[:, 1]

		# print("Set Points:")
		# print(set_points)
		# print()
		
		if set_current_count == set_target_count:
			return set_points
		
		else:
			# generate new linspace for interpolation
			x_new = np.linspace(x_coords[0], x_coords[-1], set_target_count)
			# y_new = np.interp(x_new, x_coords, y_coords)
			regression_coeffs = np.polyfit(x_coords, y_coords, 3)
			y_new = np.polyval(regression_coeffs, x_new)

			# print(x_new)
			# print(y_new)
			# print()

			return np.stack((x_new, y_new), axis=-1)
		
	# half on top and half on bottom
	section_target_count = target_count // 2


	# print()
	# print()
	# print("upper_interpolated")
	# print(upper_points)
	# print("lower_interpolated")
	# print(lower_points)
	# print()
	# print()

	# add or remove top and bottom
	upper_interpolated = interpolate_set(upper_points, section_target_count)
	lower_interpolated = interpolate_set(lower_points, section_target_count)

	# print(points_array)

	# combine the arrays
	return np.vstack((upper_interpolated, lower_interpolated))



# run the program
if __name__ == "__main__":
	
	create_and_move()
 
	# each of the potential directories
	dir_list = os.listdir("./Example Data")

	# check each of the potential files
	for potential_file in dir_list:

		print(f"Simulating {potential_file}")

		# run the simulation
		pot_dir = os.path.join("./Example Data", potential_file)

		# reorder the old points
		reformat_airfoil_data(os.path.join(pot_dir, f"{potential_file}.dat"))

		# get the new interpolated points
		# old_points = get_points_from_dat_file( os.path.join(pot_dir, f"{potential_file}_reformatted.dat"))
		# t_points = interpolate_or_decimate_points(old_points)

		# print(t_points)

		# save the new points to the reformatted files
		# reformat_airfoil_data(os.path.join(pot_dir, f"{potential_file}_reformatted.dat"), new_points = t_points, change_file=True)



