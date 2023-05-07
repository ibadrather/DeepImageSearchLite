import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from DeepSearchLite.DeepSearchLite import LoadData, SearchSetup
from Example.FeatureExtractor import CustomFeatureExtractor

os.system("clear")

MODEL_PATH = "feature_encoder.onnx"
MODE = "search"
IMAGES_DIR = "/home/ibad/Desktop/RevSearch/Car196_Combined/images/"

# Load images from a folder
image_list_all = LoadData().from_folder([IMAGES_DIR])
image_list = image_list_all[:]

print("Number of images: ", len(image_list))

feature_extractor = CustomFeatureExtractor(
    model_path=MODEL_PATH,
)

search_engine = SearchSetup(
        image_list = image_list,
        feature_extractor=feature_extractor,
        dim_reduction = None,
        image_count = None,
        metadata_dir = "cars_dataset_metadata_dir",
        feature_extractor_name = "efficientnet_onnx",
        mode=MODE,
    )

similar_n_images = search_engine.get_similar_images_list(image_path=image_list[5000], number_of_images=10)

print(similar_n_images)
