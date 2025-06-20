import os
import numpy as np
import scipy.io
import pandas as pd

class LIVE_loader:
    def __init__(self, root):
        self.img_path_map = {}
        self.root = root
        self.ref_img_names = []

        dist_types = ['jp2k', 'jpeg', 'wn', 'gblur', 'fastfading']
        dist_sizes = [227, 233, 174, 174, 174]

        # Load DMOS scores
        dmos = scipy.io.loadmat(os.path.join(root, 'dmos_realigned.mat'))
        scores = dmos['dmos_new'].astype(np.float32)
        scores = scores
        orgs = dmos['orgs'].astype(np.bool_)

        dist_ref_map = {}  # Map from distorted image path to (reference image name)

        for idx in range(len(dist_types)):
            dist_type = dist_types[idx]
            dist_filepath = os.path.join(root, dist_type)
            info_file_path = os.path.join(dist_filepath, "info.txt")
            print(dist_types[idx])

            with open(info_file_path, 'r') as file:
                for line in file:
                    if len(line) > 1:
                        info = line.strip().split()
                        dist_image_name = info[1]
                        # Adjust depth image extension if necessary
                        dist_img_path = os.path.join(dist_filepath, dist_image_name)
                        ref_img_name = info[0]
                        dist_ref_map[dist_img_path] = (ref_img_name)

        img_idx = 0

        # Second loop: Build the image path map
        for idx in range(len(dist_types)):
            dist_type = dist_types[idx]
            dist_filepath = os.path.join(root, dist_type)
            for i in range(dist_sizes[idx]):
                dist_img_name = f'img{i + 1}.bmp'
                dist_img_path = os.path.join(dist_filepath, dist_img_name)
                score = scores[0][img_idx]
                if not orgs[0][img_idx]:
                    if dist_img_path in dist_ref_map:
                        ref_img_name = dist_ref_map[dist_img_path]
                        if ref_img_name not in self.ref_img_names:
                            self.ref_img_names.append(ref_img_name)
                        self.img_path_map.setdefault(ref_img_name, []).append((dist_img_path, score))
                img_idx += 1

    def getReferenceImages(self, index):
        return [self.ref_img_names[i] for i in index]

    def load_dataset(self, index):
        ref_samples = [self.ref_img_names[i] for i in index]
        print(f"Selected reference images: {ref_samples}")
        dist_images = []
        scores = []
        for ref in ref_samples:
            for dist_img_path, score in self.img_path_map[ref]:
                dist_images.append(dist_img_path)
                scores.append(score)
        return dist_images, scores

class TID_loader():
    def __init__(self,root):
        img_path_map = {}
        dist_img_path_root = os.path.join(root,'distorted_images')
        score_file_path = os.path.join(root, "mos_with_names.txt")
        with open(score_file_path) as file:
            for line in file:
                if len(line) > 1:
                    info = line.strip().split()
                    score = float(info[0])
                    dist_img_path = os.path.join(dist_img_path_root,info[1])
                    ref_img_name = info[1].split('_')[0].lower()
                    img_path_map.setdefault(ref_img_name, []).append((dist_img_path, score))
        self.img_path_map = img_path_map
        self.root = root
        self.ref_img_names = list(img_path_map.keys())

    def getReferenceImages(self, index):
        return [(self.ref_img_names[i]) for i in index]
    
    def load_dataset(self,index):
        ref_samples = [(self.ref_img_names[i]) for i in index]
        dist_images = []
        scores = []
        for ref in ref_samples:
            for dist_img_path, score in self.img_path_map[ref]:
                dist_images.append(dist_img_path)
                scores.append(score)
        return dist_images, scores


class KADID10k_loader():
    def __init__(self, root):
        self.root = root
        self.img_path_map = {}
        self.ref_img_names = []
        
        # Load DMOS scores
        dmos_path = os.path.join(root, 'dmos.csv')
        with open(dmos_path, 'r') as file:
            next(file)  # Assuming there's a header and skipping it
            for line in file:
                parts = line.strip().split(',')
                distorted_image_name = parts[0].strip()
                ref_image_name = parts[1].strip()
                score = float(parts[2].strip())
                distorted_image_path = os.path.join(root, 'images', distorted_image_name)
                if ref_image_name not in self.ref_img_names:
                    self.ref_img_names.append(ref_image_name)
                
                self.img_path_map.setdefault(ref_image_name, []).append((distorted_image_path, score))

    def getReferenceImages(self, index):
        return [(self.ref_img_names[i]) for i in index]
    
    def load_dataset(self, index):
        ref_samples = [self.ref_img_names[i] for i in index]
        dist_images = []
        scores = []
        for ref in ref_samples:
            for dist_img_path, score in self.img_path_map[ref]:
                dist_images.append(dist_img_path)
                scores.append(score)
        return dist_images, scores  

class CSIQ_loader():
    def __init__(self, root):
        self.root = root
        self.img_path_map = {}
        self.ref_img_names = []

        # Path to distorted images and label file
        distorted_images_path = os.path.join(root, 'dst_imgs_all')
        label_file_path = os.path.join(root, 'csiq_label.txt')

        # Read the label file and store image paths and scores
        with open(label_file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    image_name, dmos = parts
                    score = float(dmos)
                    distorted_image_path = os.path.join(distorted_images_path, image_name)
                    ref_image_name = image_name.split('.')[0]
                    if ref_image_name not in self.ref_img_names:
                        self.ref_img_names.append(ref_image_name)
                    
                    self.img_path_map.setdefault(ref_image_name, []).append((distorted_image_path, score))

        # Path to reference images
        self.src_images_path = os.path.join(root, 'src_imgs')

    def getReferenceImages(self, index):
        return [(self.ref_img_names[i]) for i in index]
    
    def load_dataset(self, index):
        ref_samples = [self.ref_img_names[i] for i in index]
        dist_images = []
        scores = []
        for ref in ref_samples:
            for dist_img_path, score in self.img_path_map[ref]:
                dist_images.append(dist_img_path)
                scores.append(score)
        return dist_images, scores

class LIVECHALLENGE_loader():
    def __init__(self,root):
        img_path_map = {}
        imagenames = scipy.io.loadmat(os.path.join(root, 'Data', 'AllImages_release.mat'))
        imagenames = imagenames['AllImages_release']
        imagenames = imagenames[7:1169]
        mos = scipy.io.loadmat(os.path.join(root, 'Data', 'AllMOS_release.mat'))
        scores = mos['AllMOS_release'].astype(np.float32)  
        scores = scores[0][7:1169]
        for idx in range(len(imagenames)):
            dist_img_path = os.path.join(root,'Images/Images', imagenames[idx][0][0])
            score = scores[idx]
            img_path_map.setdefault(imagenames[idx][0][0], []).append((dist_img_path, score))
        self.img_path_map = img_path_map
        self.root = root
        self.ref_img_names = list(img_path_map.keys())

    def getReferenceImages(self, index):
        return [(self.ref_img_names[i]) for i in index]
    
    def load_dataset(self,index):
        ref_samples = [(self.ref_img_names[i]) for i in index]
        dist_images = []
        scores = []
        for ref in ref_samples:
            for dist_img_path, score in self.img_path_map[ref]:
                dist_images.append(dist_img_path)
                scores.append(score)

        return dist_images, scores

class KONIQ_loader():
    def __init__(self,root):
        img_path_map = {}
        csv_path = os.path.join(root, 'koniq10k_scores_and_distributions.csv')
        image_dir = os.path.join(root, '1024x768')

        self.df = pd.read_csv(csv_path)
        
        # Extract 'image_name' and 'MOS_zscore'
        if 'image_name' not in self.df.columns or 'MOS_zscore' not in self.df.columns:
            raise ValueError("CSV file must contain 'image_name' and 'MOS_zscore' columns.")
        
        self.df = self.df[['image_name', 'MOS_zscore']]

        for idx, row in self.df.iterrows():
            image_name = row['image_name']
            score = row['MOS_zscore']
            dist_img_path = os.path.join(image_dir, image_name)
            img_path_map.setdefault(image_name, []).append((dist_img_path, score))
        self.img_path_map = img_path_map
        self.root = root
        self.ref_img_names = list(img_path_map.keys())

    def getReferenceImages(self, index):
        return [(self.ref_img_names[i]) for i in index]
    
    def load_dataset(self,index):
        ref_samples = [(self.ref_img_names[i]) for i in index]
        dist_images = []
        scores = []
        for ref in ref_samples:
            for dist_img_path, score in self.img_path_map[ref]:
                dist_images.append(dist_img_path)
                scores.append(score)
        return dist_images, scores

class FLIVE_loader():
    def __init__(self,root):
        img_path_map = {}
        csv_path = os.path.join(root, 'labels_image.csv')
        image_dir = os.path.join(root, 'database')

        self.df = pd.read_csv(csv_path)
        
        if 'name' not in self.df.columns or 'mos' not in self.df.columns:
            raise ValueError("CSV file must contain 'image_name' and 'MOS_zscore' columns.")
        
        self.df = self.df[['name', 'mos']]

        for idx, row in self.df.iterrows():
            image_name = row['name']
            score = row['mos']
            dist_img_path = os.path.join(image_dir, image_name)
            img_path_map.setdefault(image_name, []).append((dist_img_path, score))
        self.img_path_map = img_path_map
        self.root = root
        self.ref_img_names = list(img_path_map.keys())

    def getReferenceImages(self, index):
        return "Not Supported"
    
    def load_dataset(self,index):
        ref_samples = [(self.ref_img_names[i]) for i in index]
        dist_images = []
        scores = []
        for ref in ref_samples:
            for dist_img_path, score in self.img_path_map[ref]:
                dist_images.append(dist_img_path)
                scores.append(score)
        print(len(dist_images))
        return dist_images, scores

class SPAQ_loader():
    def __init__(self,root):
        img_path_map = {}
        annotation_path = os.path.join(root, 'Annotations/MOS and Image attribute scores.xlsx')
        image_dir = os.path.join(root, 'TestImage')

        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.df = pd.read_excel(annotation_path, engine='openpyxl')
        
        if 'Image name' not in self.df.columns or 'MOS' not in self.df.columns:
            raise ValueError("CSV file must contain 'image_name' and 'MOS_zscore' columns.")
        
        self.df = self.df[['Image name', 'MOS']]

        for img in self.image_files:
            img_id = img
            img_path = os.path.join(image_dir, img)
            score = self.df.loc[self.df['Image name'] == img_id, 'MOS'].values[0]
            img_path_map.setdefault(img, []).append((img_path, score))
        self.img_path_map = img_path_map
        self.root = root
        self.ref_img_names = list(img_path_map.keys())

    def getReferenceImages(self, index):
        return [(self.ref_img_names[i]) for i in index]
    
    def load_dataset(self,index):
        ref_samples = [(self.ref_img_names[i]) for i in index]
        dist_images = []
        scores = []
        for ref in ref_samples:
            for dist_img_path, score in self.img_path_map[ref]:
                dist_images.append(dist_img_path)
                scores.append(score)
        print(len(dist_images))
        return dist_images, scores
