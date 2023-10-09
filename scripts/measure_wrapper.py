def measure_apbs(dir_path):
    import os
    import numpy as np
    import skimage
    import csv

    # Conversion factor: 1 px = 2.16 nm, so 1 px^2 = 2.16^2 nm^2
    conversion_factor = (2.16 ** 2)

    # Minimum blob area threshold (in nm^2)
    min_blob_area = 300  # Adjust this threshold as needed

    # Create a CSV file to write the area results
    area_csv_filename = 'blob_areas.csv'
    with open(area_csv_filename, 'w', newline='') as area_csvfile:
        area_writer = csv.writer(area_csvfile)

        # Create a CSV file to write the number of bodies results
        bodies_csv_filename = 'num_bodies.csv'
        with open(bodies_csv_filename, 'w', newline='') as bodies_csvfile:
            bodies_writer = csv.writer(bodies_csvfile)

            # List all PNG files in the directory
            mask_files = [f for f in os.listdir(dir_path) if f.endswith('.png') or f.endswith('.tif')]

            for mask_file in mask_files:
                # Load mask image
                mask_path = os.path.join(dir_path, mask_file)
                mask = skimage.io.imread(mask_path, as_gray=True)  # Load as grayscale

                # Perform image segmentation
                labeled_mask, num_labels = skimage.measure.label(mask, connectivity=2, return_num=True)

                # Count blobs and calculate area
                unique_labels = np.unique(labeled_mask)
                blob_areas = []

                for label in unique_labels:
                    if label != 0:  # Exclude background
                        blob_pixels = np.sum(labeled_mask == label)
                        blob_area = blob_pixels * conversion_factor
                        if blob_area >= min_blob_area:
                            blob_areas.append(round(blob_area, 2))  # Round the area to two decimal places

                # Sort blob areas in descending order
                blob_areas.sort(reverse=True)

                # Write image name and blob areas to area CSV
                for area in blob_areas:
                    area_writer.writerow([mask_file, area])

                # Write image name and number of bodies to bodies CSV
                bodies_writer.writerow([mask_file, len(blob_areas)])

    print("Blob areas saved to:", area_csv_filename)
    print("Number of bodies saved to:", bodies_csv_filename)
