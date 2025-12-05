from pathlib import Path

def main():
    data_directory = Path("rfw_dataset")

    with Path.open(data_directory / "train_labels.csv", "w") as f_train, \
         Path.open(data_directory / "test_labels.csv", "w") as f_test:
        (data_directory / "train").mkdir(exist_ok=True)
        f_train.write("file,person,race\n")
        (data_directory / "test").mkdir(exist_ok=True)
        f_test.write("file,person,race\n")

        for race_folder in (data_directory / "data").iterdir():
            race = race_folder.name

            for i, folder in enumerate(race_folder.iterdir()):
                curr_images = tuple(folder.iterdir())

                if len(curr_images) < 2:
                    continue
                    
                for image in curr_images[:-1]:
                    file_name = f"{race}_{image.name}"
                    image.rename(data_directory / f"train/{file_name}")
                    f_train.write(f"{file_name},{folder.name},{race}\n")

                file_name = f"{race}_{curr_images[-1].name}"
                curr_images[-1].rename(data_directory / f"test/{file_name}")
                f_test.write(f"{file_name},{folder.name},{race}\n")

if __name__ == "__main__":
    main()