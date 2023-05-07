import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
import faiss
from typing import List, Union, Dict, Callable, Optional


def image_data_with_features_pkl(metadata_dir, model_name):
    data_dir = os.path.join(metadata_dir, f"{model_name}")

    # Create the directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)

    image_data_with_features_pkl = os.path.join(data_dir, "image_data_features.pkl")
    return image_data_with_features_pkl


def image_features_vectors_idx(metadata_dir, model_name):
    data_dir = os.path.join(metadata_dir, f"{model_name}")

    # Create the directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)

    image_features_vectors_idx = os.path.join(data_dir, "image_features_vectors.idx")
    return image_features_vectors_idx


class LoadData:
    """A class for loading data from single/multiple folders or a CSV file"""

    def __init__(self):
        """
        Initializes an instance of LoadData class
        """
        pass

    def from_folder(self, folder_list: list):
        """
        Adds images from the specified folders to the image_list.

        Parameters:
        -----------
        folder_list : list
            A list of paths to the folders containing images to be added to the image_list.
        """
        self.folder_list = folder_list
        image_path = []
        for folder in self.folder_list:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                        image_path.append(os.path.join(root, file))
        return image_path

    def from_csv(self, csv_file_path: str, images_column_name: str):
        """
        Adds images from the specified column of a CSV file to the image_list.

        Parameters:
        -----------
        csv_file_path : str
            The path to the CSV file.
        images_column_name : str
            The name of the column containing the paths to the images to be added to the image_list.
        """
        self.csv_file_path = csv_file_path
        self.images_column_name = images_column_name
        return pd.read_csv(self.csv_file_path)[self.images_column_name].to_list()


class SearchSetup:
    """A class for setting up and running image similarity search."""

    def __init__(
        self,
        image_list: list,
        feature_extractor: Optional[Callable] = None,
        dim_reduction: Optional[Callable] = None,
        image_count: Optional[int] = None,
        metadata_dir: Optional[str] = "metadata_dir",
        feature_extractor_name: Optional[str] = "feature_extractor",
        mode: str = "index",
    ):
        """
        Parameters:
        -----------
        image_list : list
            A list of images to be indexed and searched.
        feature_extractor : Callable, optional
            Custom model for feature extraction (default=None).
        dim_reduction : Callable, optional
            Custom dimensionality reduction function (default=None).
        image_count : int, optional
            The number of images to be indexed and searched. If None, all images in the image_list will be used (default=None).
        metadata_dir : str, optional
            The directory to store metadata files (default="metadata_dir").
        feature_extractor_name : str, optional
            Name of the custom feature extractor (default="feature_extractor").
        mode : str, optional
            The mode to run the search in. Can be either "index" or "search" (default="index").
        """
        self.image_data = pd.DataFrame()
        self.model_name = feature_extractor_name

        self.mode = mode

        self.image_list = (
            image_list[:image_count] if image_count is not None else image_list
        )

        self.feature_extractor = feature_extractor
        self.dim_reduction = dim_reduction

        # Create metadata directory
        self.metadata_dir = metadata_dir
        os.makedirs(self.metadata_dir, exist_ok=True)

        if self.mode == "index":
            self.run_index()
        elif self.mode == "search":
            self.load_metadata()
        else:
            raise ValueError("Invalid mode. Must be 'index' or 'search'.")

    def _extract(self, img: Image.Image) -> np.ndarray:
        """Extract features from the image."""
        # Resize and convert the image
        img = img.resize((224, 224))
        img = img.convert("RGB")

        feature = self.feature_extractor(img)

        # Normalize the feature vector
        feature = feature.flatten()

        # Dimensionality reduction
        if self.dim_reduction is not None:
            feature = self.dim_reduction(feature)

        return feature / np.linalg.norm(feature)

    def _get_feature(self, image_data: List[str]) -> List[Union[np.ndarray, None]]:
        self.image_data = image_data
        features = []
        for img_path in tqdm(self.image_data):  # Iterate through images
            # Extract features from the image
            try:
                feature = self._extract(img=Image.open(img_path))
                features.append(feature)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                features.append(None)
                continue
        return features

    def _start_feature_extraction(self) -> pd.DataFrame:
        image_data = pd.DataFrame()
        image_data["images_paths"] = self.image_list
        f_data = self._get_feature(self.image_list)
        image_data["features"] = f_data
        image_data = image_data.dropna().reset_index(drop=True)

        image_data.to_pickle(
            image_data_with_features_pkl(self.metadata_dir, self.model_name)
        )

        print(
            f"\033[94m Image Meta Information Saved: [os.path.join(self.metadata_dir, self.model_name, 'image_data_features.pkl')]"
        )
        return image_data

    def _start_indexing(self, image_data: pd.DataFrame) -> None:
        self.image_data = image_data
        d = len(image_data["features"][0])  # Length of item vector that will be indexed
        self.d = d
        index = faiss.IndexFlatL2(d)
        features_matrix = np.vstack(image_data["features"].values).astype(np.float32)
        index.add(features_matrix)  # Add the features matrix to the index
        faiss.write_index(
            index, image_features_vectors_idx(self.metadata_dir, self.model_name)
        )

        print(
            "\033[94m Saved The Indexed File:"
            + f"[os.path.join(self.metadata_dir, self.model_name, 'image_features_vectors.idx')]"
        )

    def run_index(self) -> None:
        """
        Indexes the images in the image_list and creates an index file for fast similarity search.
        """
        if len(os.listdir(self.metadata_dir)) == 0:
            data = self._start_feature_extraction()
            self._start_indexing(data)
        else:
            user_input = input(
                "\033[91m Metadata and Features are already present, Do you want Extract Again? Enter yes or no: "
            )

            if user_input.lower() == "yes":
                data = self._start_feature_extraction()
                self._start_indexing(data)
            else:
                print("\033[93m Meta data already Present, Please Apply Search!")
                print(os.listdir(self.metadata_dir))
        self.image_data = pd.read_pickle(
            image_data_with_features_pkl(self.metadata_dir, self.model_name)
        )
        self.f = len(self.image_data["features"][0])

    def add_images_to_index(self, new_image_paths: List[str]) -> None:
        """
        Adds new images to the existing index.

        Parameters:
        -----------
        new_image_paths : list
            A list of paths to the new images to be added to the index.
        """
        # Load existing metadata and index
        self.image_data = pd.read_pickle(
            image_data_with_features_pkl(self.metadata_dir, self.model_name)
        )
        index = faiss.read_index(
            image_features_vectors_idx(self.metadata_dir, self.model_name)
        )

        for new_image_path in tqdm(new_image_paths):
            # Extract features from the new image
            try:
                img = Image.open(new_image_path)
                feature = self._extract(img)
            except Exception as e:
                print(f"\033[91m Error extracting features from the new image: {e}")
                continue

            # Add the new image to the metadata
            new_metadata = pd.DataFrame(
                {"images_paths": [new_image_path], "features": [feature]}
            )
            self.image_data = pd.concat(
                [self.image_data, new_metadata], axis=0, ignore_index=True
            )

            # Add the new image to the index
            index.add(np.array([feature], dtype=np.float32))

        # Save the updated metadata and index
        self.image_data.to_pickle(
            image_data_with_features_pkl(self.metadata_dir, self.model_name)
        )
        faiss.write_index(
            index, image_features_vectors_idx(self.metadata_dir, self.model_name)
        )

        print(f"\033[92m New images added to the index: {len(new_image_paths)}")

    def _search_by_vector(self, v: np.ndarray, n: int) -> Dict[int, str]:
        index = faiss.read_index(
            image_features_vectors_idx(self.metadata_dir, self.model_name)
        )
        D, I = index.search(np.array([v], dtype=np.float32), n)
        return dict(zip(I[0], self.image_data.iloc[I[0]]["images_paths"].to_list()))

    def _get_query_vector(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path)
        query_vector = self._extract(img)
        return query_vector

    def get_similar_images(
        self, image_path: str, number_of_images: int = 10
    ) -> Dict[int, str]:
        """
        Returns the most similar images to a given query image according to the indexed image features.

        Parameters:
        -----------
        image_path : str
            The path to the query image.
        number_of_images : int, optional (default=10)
            The number of most similar images to the query image to be returned.
        """
        query_vector = self._get_query_vector(image_path)
        img_dict = self._search_by_vector(query_vector, number_of_images)
        return img_dict

    def get_similar_images_list(
        self, image_path: str, number_of_images: int = 10
    ) -> List[str]:
        """
        Returns the most similar images to a given query image according to the indexed image features.

        Parameters:
        -----------
        image_path : str
            The path to the query image.
        number_of_images : int, optional (default=10)
            The number of most similar images to the query image to be returned.
        """
        img_dict = self.get_similar_images(image_path, number_of_images)
        similar_n_images =  list(img_dict.values())
        similar_n_images_names = [os.path.basename(image_path) for image_path in similar_n_images]
        return similar_n_images_names

    def get_image_metadata_file(self) -> pd.DataFrame:
        """
        Returns the metadata file containing information about the indexed images.

        Returns:
        --------
        DataFrame
            The Panda DataFrame of the metadata file.
        """
        image_data = pd.read_pickle(
            image_data_with_features_pkl(self.metadata_dir, self.model_name)
        )
        return image_data

    def load_metadata(self):
        """Loads the metadata and index for search mode."""
        self.image_data = pd.read_pickle(
            image_data_with_features_pkl(self.metadata_dir, self.model_name)
        )
        self.f = len(self.image_data["features"][0])


    def _get_query_vector_from_image(self, image: Image) -> np.ndarray:
        query_vector = self._extract(image)
        return query_vector
    
    def get_similar_images_from_image(
        self, image: Image, number_of_images: int = 10
    ) -> Dict[int, str]:
        """
        Returns the most similar images to a given query image according to the indexed image features.

        Parameters:
        -----------
        image : Image
            The query image.
        number_of_images : int, optional (default=10)
            The number of most similar images to the query image to be returned.
        """
        query_vector = self._get_query_vector_from_image(image)
        img_dict = self._search_by_vector(query_vector, number_of_images)
        return img_dict
    
    def get_similar_images_list_from_image(
        self, image: Image, number_of_images: int = 10
    ) -> List[str]:
        """
        Returns the most similar images to a given query image according to the indexed image features.

        Parameters:
        -----------
        image : Image
            The query image.
        number_of_images : int, optional (default=10)
            The number of most similar images to the query image to be returned.
        """
        img_dict = self.get_similar_images_from_image(image, number_of_images)
        similar_n_images =  list(img_dict.values())
        similar_n_images_names = [os.path.basename(image_path) for image_path in similar_n_images]
        return similar_n_images_names
