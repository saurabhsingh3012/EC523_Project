import os
import csv
import shutil

# Define the path to the directory containing the .wav files
dir_path = "xc_ny_soundscape_seperated_copy"

# Define the path to the CSV file containing the index and names
csv_path = "xc_ny_soundscape/ny_metadata.csv"

# Load the index and name data from the CSV file
with open(csv_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    # Skip the header row
    next(csv_reader)
    # Create a dictionary mapping index to name
    index_to_name = {int(row[0]): row[1] + ' ' + row[2] + '_' + row[5] for row in csv_reader}

# Iterate over the files in the directory
for filename in os.listdir(dir_path):
    # Split the file extension from the filename
    name, extension = os.path.splitext(filename)
    try:
		# Get the index from the filename
        index = int(name)
    except ValueError:
        # Skip any filenames that can't be converted to integers
        continue
        
    
    # Check if the index is in the dictionary
    if index in index_to_name:
        # Rename the file using the corresponding name
        new_name = index_to_name[index]
        new_path = os.path.join(dir_path, new_name)
        old_path = os.path.join(dir_path, filename)
        if os.path.exists(new_path):
            # Copy the file to a new location with a different name
            temp_path = os.path.join(dir_path, 'temp_' + new_name)
            shutil.copy2(old_path, temp_path)

            # Delete the original file
            os.remove(old_path)

            # Rename the copied file to the desired name
            os.rename(temp_path, new_path)
        else:
            os.replace(old_path, new_path)
