import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

# File paths
combined_dataset_path = r'C:\Users\mgummuluri\Work\Tasks\YOLOv10\combined_dataset'
gcp_csv_path = r'C:\Users\mgummuluri\Work\Tasks\YOLOv10\gcp_data.csv'
output_xml_path = r'C:\Users\mgummuluri\Work\Tasks\YOLOv10\test_output.xml'

# Load GCP data
gcp_data = pd.read_csv(gcp_csv_path)

# Paths for images and labels
images_path = os.path.join(combined_dataset_path, "images")
labels_path = os.path.join(combined_dataset_path, "labels")

# Initialize XML structure
root = ET.Element("SurveysData")

# Add Spatial Reference System
srs = ET.SubElement(root, "SpatialReferenceSystems")
srs_1 = ET.SubElement(srs, "SRS")
ET.SubElement(srs_1, "Id").text = "1"
ET.SubElement(srs_1, "Name").text = "WGS 84"
ET.SubElement(srs_1, "Definition").text = "EPSG:4326"

# Add Control Points
control_points = ET.SubElement(root, "ControlPoints")

# Process GCP data
for _, gcp_row in gcp_data.iterrows():
    gcp_id = int(gcp_row['ID'])
    gcp_name = f"GCP_{gcp_id}"
    
    # Create <ControlPoint> element
    control_point = ET.SubElement(control_points, "ControlPoint")
    ET.SubElement(control_point, "Id").text = str(gcp_id)
    ET.SubElement(control_point, "SRSId").text = "1"
    ET.SubElement(control_point, "Name").text = gcp_name
    ET.SubElement(control_point, "Category").text = "Full"

    # Add GCP position
    position = ET.SubElement(control_point, "Position")
    ET.SubElement(position, "x").text = str(gcp_row['x'])
    ET.SubElement(position, "y").text = str(gcp_row['y'])
    ET.SubElement(position, "z").text = str(gcp_row['z'])

    ET.SubElement(control_point, "HorizontalAccuracy").text = "0.01"
    ET.SubElement(control_point, "VerticalAccuracy").text = "0.01"
    ET.SubElement(control_point, "CheckPoint").text = "false"

    # Add measurements for each label corresponding to the GCP ID
    for label_file in os.listdir(labels_path):
        label_path = os.path.join(labels_path, label_file)
        with open(label_path, 'r') as file:
            annotations = file.readlines()

        for annotation in annotations:
            fields = annotation.strip().split()
            
            # Extract class ID and coordinates
            class_id = int(fields[0])
            if class_id != gcp_id:
                continue

            # Parse corner coordinates
            corners = np.array(list(map(float, fields[1:])))
            corners = corners.reshape(-1, 2)  # Reshape into pairs (x, y)

            # Calculate center of the bounding box
            center_x = np.mean(corners[:, 0])
            center_y = np.mean(corners[:, 1])

            # Add measurement to XML
            image_name = label_file.replace(".txt", ".jpg")
            measurement = ET.SubElement(control_point, "Measurement")
            ET.SubElement(measurement, "PhotoId").text = "0"
            ET.SubElement(measurement, "ImagePath").text = os.path.join(images_path, image_name)
            ET.SubElement(measurement, "x").text = str(center_x)
            ET.SubElement(measurement, "y").text = str(center_y)

# Write XML to file with proper formatting
tree = ET.ElementTree(root)
with open(output_xml_path, "wb") as file:
    tree.write(file, encoding="utf-8", xml_declaration=True)

print(f"XML file successfully generated: {output_xml_path}")
