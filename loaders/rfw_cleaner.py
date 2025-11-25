import pandas as pd
import os, shutil

classes = ['African', 'Asian', 'Caucasian', 'Indian']
class_count = 4

merged_image_path = "./dataset/RFW/all/"
# merged_csv_path = "./dataset/RFW/dataset_rfw_all.csv"
merged_csv_path = "./dataset/RFW/rfw_cleaned_labels.csv"

train_image_path = "./dataset/RFW/train/"
test_image_path = "./dataset/RFW/test/"
train_label_path = "./dataset/rfw_train_labels.csv"
test_label_path = "./dataset/rfw_test_labels.csv"

def csv_cleanup(df, image_path, output_path):
    # delete labels without images
    # get list of available images
    available_images = os.listdir(image_path)
    initial_size = len(df)
    df = df[df['file'].isin(available_images)].reset_index(drop=True)
    final_size = len(df)
    print(f"Cleaned RFW testing labels: removed {initial_size - final_size} entries without corresponding images.")
    df.to_csv(output_path, index=False) 

    #verify all images have labels
    image_files = set(os.listdir(image_path))
    print(f"Total images in {image_path}: {len(image_files)}")
    label_files = set(df['file'].tolist())
    # list duplicate images in labels
    duplicates = df['file'][df['file'].duplicated()].unique()
    if len(duplicates) > 0:
        print(f"Warning: Found {len(duplicates)} duplicate image entries in labels: {duplicates}")
    #delete duplicates that show up later in the csv
    df = df.drop_duplicates(subset=['file'], keep='first').reset_index(drop=True)
    label_files = set(df['file'].tolist())
    
    print(f"Total labeled images in CSV: {len(label_files)}")
    unlabeled_images = image_files - label_files
    if len(unlabeled_images) == 0:
        print("All images have corresponding labels.")
    else:
        print(f"Found {len(unlabeled_images)} images without labels.")
        print("Unlabeled images:", unlabeled_images)

def merger(train_df, test_df, output_path):
    #merge train and test dataframes and save to output_path
    df = pd.concat([train_df, test_df], ignore_index=True)
    csv_cleanup(df, train_image_path, output_path)

def move_images(df, source_path, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for idx, row in df.iterrows():
        src_file = os.path.join(source_path, row['file'])
        dest_file = os.path.join(dest_path, row['file'])
        if not os.path.exists(src_file):
            print(f"Warning: source file {src_file} does not exist.")
        else:
            shutil.copy(src_file, dest_file)

df = pd.read_csv(merged_csv_path)
# csv_cleanup(df, merged_image_path, new_csv_path)

full_df = pd.read_csv(merged_csv_path)
full_df = full_df.sort_values(by=['race']).reset_index(drop=True)
#print class distribution
print("Full dataset class distribution:")
print(full_df["race"].value_counts())
#verify all images exist
img_list = os.listdir(merged_image_path)
image_files = set(img_list)
label_list = full_df['file'].tolist()
label_files = set(label_list)
print(f"Total images in merged folder: {len(image_files)}, total labels in csv: {len(label_files)}")
unlabeled_images = label_files - image_files
if len(unlabeled_images) == 0:
    print("All labels have corresponding images.")
else:
    print(f"Found {len(unlabeled_images)} labels without images.")

#check for images with only one person
person_counts = full_df['person'].value_counts()
single_image_persons = person_counts[person_counts == 1].index.tolist()
if len(single_image_persons) > 0:
    print(f"Warning: Found {len(single_image_persons)} persons with only one image")
    print("Examples:", single_image_persons[:5])

# take one image from all ['person']s with more than one image for testing, rest for training
test_indices = []

race_df = full_df.copy()
person_counts = race_df['person'].value_counts()
multi_image_persons = person_counts[person_counts > 1].index.tolist()
test_persons = multi_image_persons
for person in test_persons:
    person_df = race_df[race_df['person'] == person]
    test_idx = person_df.index[0]  # take the first image of the person
    test_indices.append(test_idx)
test_df = full_df.loc[test_indices].reset_index(drop=True)
train_df = full_df.drop(index=test_indices).reset_index(drop=True)
#sort by race, person
test_df = test_df.sort_values(by=['race', 'person']).reset_index(drop=True)
train_df = train_df.sort_values(by=['race', 'person']).reset_index(drop=True)

# verify all test persons are in train set
train_persons = set(train_df['person'].tolist())
test_persons = set(test_df['person'].tolist())
missing_persons = test_persons - train_persons
if len(missing_persons) == 0:
    print("All test persons are present in training set.")
    # set same index for each person in train and test set
    person_to_index = {person: idx for idx, person in enumerate(sorted(train_persons))}
    train_df['label_idx'] = train_df['person'].map(person_to_index)
    test_df['label_idx'] = test_df['person'].map(person_to_index)
    train_df.to_csv(train_label_path, index=False)
    test_df.to_csv(test_label_path, index=False)
    move_images(train_df, merged_image_path, train_image_path)
    move_images(test_df, merged_image_path, test_image_path)
else:
    print(f"Warning: Found {len(missing_persons)} persons in test set not present in training set.")
    print("Examples:", list(missing_persons)[:5])

print(f"Final training set size: {len(train_df)}, testing set size: {len(test_df)}")
print(f"Number of unique persons in training set: {train_df['person'].nunique()}, testing set: {test_df['person'].nunique()}")

print("Final training set number of persons per race, total images per race:")
print(train_df.groupby('race')['person'].nunique(), train_df['race'].value_counts())
print("Final testing set number of persons per race:")
print(test_df.groupby('race')['person'].nunique(), test_df['race'].value_counts())