import csv
from pathlib import Path

# Define the path to the directory containing the .wav files
dir_path = Path("xc_ny_soundscape_seperated_copy")

# Define the path to the CSV file containing the index and names
csv_path = Path("xc_ny_soundscape/ny_metadata.csv")

# Load the index and name data from the CSV file
with open(csv_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    # Skip the header row
    next(csv_reader)
    # Create a dictionary mapping index to name
    index_to_name = {int(row[0]): row[1] + ' ' + row[2] + '_' + row[5] for row in csv_reader}

# Iterate over the files in the directory
for file_path in dir_path.glob("*.wav"):
    # Get the index from the filename
    index = int(file_path.stem)

   # Check if the index is in the dictionary
    if index in index_to_name:
        # Rename the file using the corresponding name
        new_name = index_to_name[index]
       # old_path = os.path.join(dir_path, dir_name)
        #new_path = os.path.join(dir_path, new_name)
        subprocess.run(["mv", dir_name, new_name])
