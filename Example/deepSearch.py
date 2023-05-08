import os
import sys
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from DeepSearchLite.DeepSearchLite import LoadData, SearchSetup
from Example.FeatureExtractor import CustomFeatureExtractor

os.system("clear")

MODEL_PATH = "Example/feature_encoder.onnx"
MODE = "search"
IMAGES_DIR = "/home/ibad/Desktop/RevSearch/Car196_Combined/images/"

# Load images from a folder
image_list_all = LoadData(data_type="image").from_folder([IMAGES_DIR])
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
        metadata_dir = "Example/cars_dataset_metadata_dir",
        feature_extractor_name = "efficientnet_onnx",
        mode=MODE,
    )

car_image_path="car.jpg"


# Search by image
car_image = Image.open(car_image_path).resize((224, 224))
similar_n_images = search_engine.get_similar_items(
    item=car_image,
    number_of_items=10,
    return_paths=True,
)

print("Similar Images by inputing an image: ", len(similar_n_images))

# Search by path
similar_n_images = search_engine.get_similar_items(
    item=car_image_path,
    number_of_items=10,
    return_paths=True,
)

print("Similar Images by inputing a path: ", len(similar_n_images))

