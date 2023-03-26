### Data labeling
"""
From the dataset file 'label_names.txt':

* 1 background 1
* 2 facade 2
* 3 window 10
* 4 door 5
* 5 cornice 11
* 6 sill 3
* 7 balcony 4
* 8 blind 6
* 9 deco 8
* 10 molding 7
* 11 pillar 12
* 12 shop 9

"""
import os
import csv
from PIL import Image

# path to the dataset directory
dataset_dir = "C:\\Users\\HOME\\PycharmProjects\\CV_BuildingAnalytics\\base"

# list of image formats to process
image_formats = ['.jpg', '.png', '.html']

# create the csv file and write the header row
csv_file = open('labels.csv', 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['filename', 'background', 'facade', 'window', 'door', 'cornice', 'sill', 'balcony', 'blind', 'deco', 'molding', 'pillar', 'shop'])

# iterate over the files in the dataset directory
for filename in os.listdir(dataset_dir):
    # check if the file is an image
    if os.path.splitext(filename)[-1].lower() in image_formats:
        # open the image and get its size
        img_path = os.path.join(dataset_dir, filename)
        img = Image.open(img_path)
        width, height = img.size

        # write the row to the csv file
        row = [filename]
        for label in range(1, 13):
            if str(label) in filename:
                row.append('0')
            else:
                row.append('1')
        writer.writerow(row)

csv_file.close()