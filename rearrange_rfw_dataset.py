from pathlib import Path

def main():
    data_directory = Path("rfw_dataset")

    with Path.open(data_directory / "train_labels.csv", "w") as f_train, \
         Path.open(data_directory / "val_labels.csv", "w") as f_val, \
         Path.open(data_directory / "test_labels.csv", "w") as f_test:
        files = {"train": f_train, "val": f_val, "test": f_test}

        for split in ("train", "val", "test"):
            (data_directory / split).mkdir(exist_ok=True)
            files[split].write("file,person,race\n")

        for race_folder in (data_directory / "data").iterdir():
            race = race_folder.name
            
            folders_tuple = tuple(race_folder.iterdir())
            
            train_threshold = int(0.6 * len(folders_tuple))
            val_threshold = int(0.8 * len(folders_tuple))

            for i, folder in enumerate(folders_tuple):
                split = "train" if i < train_threshold else "val" if i < val_threshold else "test"
                f = files[split]

                for image in folder.iterdir():
                    file_name = f"{race}_{image.name}"
                    image.rename(data_directory / split / file_name)
                    f.write(f"{file_name},{folder.name},{race}\n")

if __name__ == "__main__":
    main()