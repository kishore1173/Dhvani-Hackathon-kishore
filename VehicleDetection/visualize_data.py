import os
import xml.etree.ElementTree as ET
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class DataVisualizer:
    def __init__(self, annotations_dir, selected_classes):
        self.annotations_dir = annotations_dir
        self.selected_classes = selected_classes

    def extract_info_from_xml(self, xml_file):
        """Parse XML annotation file."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            info_dict = {'bboxes': [], 'image_size': None}
            
            for elem in root:
                if elem.tag == "size":
                    w, h, d = [int(subelem.text) for subelem in elem]
                    info_dict['image_size'] = (w, h, d)
                elif elem.tag == "object":
                    bbox = {}
                    for subelem in elem:
                        if subelem.tag == "name" and subelem.text in self.selected_classes:
                            bbox["class"] = subelem.text
                        elif subelem.tag == "bndbox" and "class" in bbox:
                            for subsubelem in subelem:
                                bbox[subsubelem.tag] = int(subsubelem.text)
                    if 'class' in bbox:
                        info_dict['bboxes'].append(bbox)
            
            return info_dict if info_dict['image_size'] else None
        
        except ET.ParseError:
            print(f"Skipping corrupted XML: {xml_file}")
            return None
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            return None

    def visualize_dataset(self):
        """Generate visualizations for class counts and bounding box areas."""
        class_counts = {cls: 0 for cls in self.selected_classes}
        areas = []

        xml_files = [os.path.join(self.annotations_dir, f) for f in os.listdir(self.annotations_dir) if f.endswith('.xml')]
        for xml_file in xml_files:
            info = self.extract_info_from_xml(xml_file)
            if info:
                iw, ih, _ = info['image_size']
                for b in info['bboxes']:
                    class_counts[b['class']] += 1
                    area = (b['xmax'] - b['xmin']) * (b['ymax'] - b['ymin'])
                    areas.append(area)

        plt.figure(figsize=(8, 5))
        sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
        plt.title("Object Counts per Class")
        plt.show()

        plt.figure(figsize=(8, 5))
        sns.histplot(np.log1p(areas), bins=30, kde=True)
        plt.title("Distribution of Bounding Box Areas (log scale)")
        plt.show()
