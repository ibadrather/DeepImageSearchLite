import csv
from pathlib import Path
from typing import List


class LoadData:
    """A class for loading data from single/multiple folders or from a CSV file.

    Attributes:
        data_type (str): The type of data to load. Supported types include
            'image', 'video', 'text', etc.
    """

    def __init__(self, data_type: str = "image"):
        """
        Initializes an instance of the LoadData class.

        Parameters:
            data_type (str): The type of data to load. Defaults to 'image'.
                             Supported types: 'image', 'video', 'text', etc.
        """
        self.data_type = data_type

    def from_folder(self, folder_list: List[Path]) -> List[Path]:
        """
        Loads data files from specified folders.

        Parameters:
            folder_list (List[Path]): A list of paths to folders containing data files.

        Returns:
            List[Path]: A list of paths to the loaded data files.
        """
        data_paths: List[Path] = []
        for folder in folder_list:
            for path in folder.rglob("*"):
                if path.is_file() and self._is_supported_file(path):
                    data_paths.append(path)
        return data_paths

    def _is_supported_file(self, path: Path) -> bool:
        """
        Checks if a file has a supported extension for the current data_type.

        Parameters:
            path (Path): The path object of the file to check.

        Returns:
            bool: True if the file has a supported extension, False otherwise.
        """
        supported_extensions = {
            "image": [".png", ".jpg", ".jpeg", ".gif", ".bmp"],
            # Add more supported extensions for other data types as needed
            # "video": [".mp4", ".avi", ".mov", ".mkv"],
            # "text": [".txt", ".csv", ".tsv", ".json"],
        }

        if self.data_type not in supported_extensions:
            raise ValueError(f"Unsupported data type: {self.data_type}")

        return any(path.name.lower().endswith(ext) for ext in supported_extensions[self.data_type])

    def from_csv(self, csv_file_path: Path, items_column_name: str) -> List[str]:
        """
        Loads items from a specified column of a CSV file using Python's built-in csv module.

        Parameters:
            csv_file_path (Path): The path to the CSV file.
            items_column_name (str): The name of the column containing the item paths.

        Returns:
            List[str]: A list of item paths as strings.
        """
        items: List[str] = []
        with csv_file_path.open(newline="", mode="r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if items_column_name in row:
                    items.append(row[items_column_name])
                else:
                    raise ValueError(f"Column {items_column_name} does not exist in the CSV file")
        return items


def item_data_with_features_pkl(metadata_dir: Path, model_name: str) -> Path:
    """
    Constructs a Path object for the 'item_data_features.pkl' file within a specified model's
    directory.

    Parameters:
        metadata_dir (Path): The directory containing model metadata.
        model_name (str): The name of the model.

    Returns:
        Path: A Path object pointing to the 'item_data_features.pkl' file.
    """
    data_dir = metadata_dir / model_name

    # Create the directory if it does not exist
    data_dir.mkdir(parents=True, exist_ok=True)

    item_data_with_features_pkl = data_dir / "item_data_features.pkl"
    return item_data_with_features_pkl


def item_features_vectors_idx(metadata_dir: Path, model_name: str) -> Path:
    """
    Constructs a Path object for the 'item_features_vectors.idx' file within a specified model's
    directory.

    Parameters:
        metadata_dir (Path): The directory containing model metadata.
        model_name (str): The name of the model.

    Returns:
        Path: A Path object pointing to the 'item_features_vectors.idx' file.
    """
    data_dir = metadata_dir / model_name

    # Create the directory if it does not exist
    data_dir.mkdir(parents=True, exist_ok=True)

    item_features_vectors_idx = data_dir / "item_features_vectors.idx"
    return item_features_vectors_idx
