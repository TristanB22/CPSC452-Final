import os

import tqdm

# this program is going to go through each of the example data files and fix them


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
                # print(f"Skipping invalid line: {line}")
                pass

    return points


BASE_DIR = "./Example Data"
file_list = os.listdir(BASE_DIR)


# # go through each of the files and fix them
for file_path in tqdm.tqdm(file_list):

    try:

        points = get_points_from_dat_file(
            os.path.join(BASE_DIR, file_path, file_path + "_reformatted.dat")
        )

        # print(points)
        # print(points[-1])
        points = points + [points[0]]

        with open(
            os.path.join(
                BASE_DIR, file_path, file_path + "_reformatted_full_points.dat"
            ),
            "w",
        ) as new_file:
            for x, y in points:
                new_file.write(f"{x:.6f}\t{y:.6f}\n")

    except Exception as e:
        print(e)
        pass

# 	# create the file path
# 	file_path = os.path.join("./Example Data", file_path)

# 	try:

# 		# open the file
# 		file_contents = open(file_path, 'r').readlines()

# 		# check the lines
# 		res_lines = [file_contents[0]]

# 	except:

# 		print(f"Skipping {file_path}")
# 		continue

# 	for tl in file_contents[1:]:

# 		tl = tl.strip()

# 		try:

# 			line_components = tl.split()
# 			if "." in line_components[0] and "." in line_components[1] and len(line_components[0]) > 3 and len(line_components[1]) > 3:
# 				res_lines.append(tl)

# 		except:
# 			pass

# 	# write the file
# 	w_file = open(file_path, 'w')

# 	for tl in res_lines:
# 		w_file.write(tl + "\n")
