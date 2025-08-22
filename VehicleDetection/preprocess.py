import os
import shutil
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, input_folder, working_dir):
        random.seed(108)
        self.input_folder = input_folder
        self.working_dir = working_dir
        self.images_dir = os.path.join(working_dir, 'images')
        self.annotations_dir = os.path.join(working_dir, 'annotations')
        self.labels_dir = os.path.join(working_dir, 'labels')
        self.selected_classes = ['car', 'bus', 'truck', 'motorbike']
        self.class_name_to_id_mapping = {cls: idx for idx, cls in enumerate(self.selected_classes)}

    def setup_directories(self):
        """Create necessary directories."""
        for d in [self.working_dir, self.images_dir, self.annotations_dir, self.labels_dir]:
            os.makedirs(d, exist_ok=True)

    def copy_and_clean_dataset(self):
        """Copy images and annotations, keeping only valid pairs."""
        all_files = os.listdir(self.input_folder)
        images = [f for f in all_files if f.lower().endswith('.jpg')]
        annotations = [f for f in all_files if f.lower().endswith('.xml')]

        # Copy files
        for img in images:
            shutil.copy(os.path.join(self.input_folder, img), os.path.join(self.images_dir, img))
        for ann in annotations:
            shutil.copy(os.path.join(self.input_folder, ann), os.path.join(self.annotations_dir, ann))

        # Match valid pairs
        image_names = {os.path.splitext(img)[0] for img in os.listdir(self.images_dir)}
        ann_names = {os.path.splitext(ann)[0] for ann in os.listdir(self.annotations_dir)}

        for img in os.listdir(self.images_dir):
            if os.path.splitext(img)[0] not in ann_names:
                os.remove(os.path.join(self.images_dir, img))

        for ann in os.listdir(self.annotations_dir):
            if os.path.splitext(ann)[0] not in image_names:
                os.remove(os.path.join(self.annotations_dir, ann))

        print(f"Matched images: {len(os.listdir(self.images_dir))}")
        print(f"Matched annotations: {len(os.listdir(self.annotations_dir))}")

    def extract_info_from_xml(self, xml_file):
        """Parse XML annotation file."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            info_dict = {'bboxes': [], 'filename': None, 'image_size': None}
            
            for elem in root:
                if elem.tag == "filename":
                    info_dict['filename'] = elem.text
                elif elem.tag == "size":
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
            
            return info_dict if info_dict['filename'] else None
        
        except ET.ParseError:
            print(f"Skipping corrupted XML: {xml_file}")
            return None
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            return None

    def convert_to_yolov5(self, info_dict, txt_path):
        """Convert bounding box data to YOLOv5 format."""
        lines = []
        if not info_dict or not info_dict['bboxes']:
            open(txt_path, 'w').close()
            return
        iw, ih, _ = info_dict['image_size']
        for b in info_dict["bboxes"]:
            cls_id = self.class_name_to_id_mapping[b["class"]]
            cx = (b["xmin"] + b["xmax"]) / 2 / iw
            cy = (b["ymin"] + b["ymax"]) / 2 / ih
            w = (b["xmax"] - b["xmin"]) / iw
            h = (b["ymax"] - b["ymin"]) / ih
            lines.append(f"{cls_id} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}")
        with open(txt_path, 'w') as f:
            f.write("\n".join(lines))

    def process_annotations(self):
        """Process all XML annotations to YOLOv5 format."""
        xml_files = [os.path.join(self.annotations_dir, f) for f in os.listdir(self.annotations_dir) if f.endswith('.xml')]
        valid_image_files, valid_label_files = [], []

        for xml_file in tqdm(xml_files):
            info_dict = self.extract_info_from_xml(xml_file)
            if info_dict:
                img_name = info_dict['filename']
                img_path = os.path.join(self.images_dir, img_name)
                txt_name = img_name.replace('.jpg', '.txt')
                txt_path = os.path.join(self.labels_dir, txt_name)
                if os.path.exists(img_path):
                    self.convert_to_yolov5(info_dict, txt_path)
                    valid_image_files.append(img_path)
                    valid_label_files.append(txt_path)

        print(f"Final pairs: {len(valid_image_files)}")
        return valid_image_files, valid_label_files

    def split_dataset(self, valid_image_files, valid_label_files):
        """Split dataset into train, validation, and test sets."""
        train_imgs, temp_imgs, train_lbls, temp_lbls = train_test_split(
            valid_image_files, valid_label_files, test_size=0.2, random_state=42
        )
        val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(
            temp_imgs, temp_lbls, test_size=0.5, random_state=42
        )

        def make_dirs(base, sub):
            path = os.path.join(base, sub)
            os.makedirs(path, exist_ok=True)
            return path

        train_img_dir = make_dirs(self.images_dir, "train")
        val_img_dir = make_dirs(self.images_dir, "val")
        test_img_dir = make_dirs(self.images_dir, "test")
        train_lbl_dir = make_dirs(self.labels_dir, "train")
        val_lbl_dir = make_dirs(self.labels_dir, "val")
        test_lbl_dir = make_dirs(self.labels_dir, "test")

        def move_files(files, dest):
            for f in files:
                if not os.path.exists(f):
                    continue
                fname = os.path.basename(f)
                shutil.move(f, os.path.join(dest, fname))

        move_files(train_imgs, train_img_dir)
        move_files(val_imgs, val_img_dir)
        move_files(test_imgs, test_img_dir)
        move_files(train_lbls, train_lbl_dir)
        move_files(val_lbls, val_lbl_dir)
        move_files(test_lbls, test_lbl_dir)

        yaml_path = os.path.join(self.working_dir, 'vehicle_data.yaml')
        with open(yaml_path, 'w') as f:
            f.write(f"""
train: {train_img_dir}
val: {val_img_dir}
test: {test_img_dir}

nc: {len(self.selected_classes)}
names: {self.selected_classes}
""")
        print(f"YAML created at {yaml_path}")
        return yaml_path
