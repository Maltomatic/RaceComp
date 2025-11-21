import pandas as pd
import os, shutil

classes = ['African', 'Asian', 'Caucasian', 'Indian']
class_count = 4

src_image_path = "./dataset/rfw_src/data/"
dest_image_path = "./dataset/RFW/imgs/"

output_path = "./dataset/RFW/dataset_rfw_all.csv"

def extractor():
    #take all images from image_path/race/label/xxx.jpg and merge into /image_path/all/xxx.jpg
    data = []
    for race in classes:
        print(f"Processing race: {race}")
        rootdir = os.path.join(src_image_path, race)
        for label in os.listdir(rootdir):
            label_dir = os.path.join(rootdir, label)
            if not os.path.isdir(label_dir):
                continue
            cnt = 0
            for file in os.listdir(label_dir):
                cnt += 1
                src_file_path = os.path.join(label_dir, file)
                dest_file_path = os.path.join(dest_image_path, file)
                #move file
                os.makedirs(dest_image_path, exist_ok=True)
                if os.path.exists(dest_file_path):
                    dest_file_path = os.path.join(dest_image_path, f"(1)_{file}")
                os.rename(src_file_path, dest_file_path)
                data.append({'file': file, 'race': race, 'person': label, 'counts': cnt})
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

extractor()