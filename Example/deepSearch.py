import os
import sys
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from DeepSearchLite.DeepSearchLite import LoadData, SearchSetup
from Example.FeatureExtractor import CustomFeatureExtractor

os.system("clear")

MODEL_PATH = "Example/feature_encoder.onnx"
IMAGES_DIR = "/home/ibad/Desktop/RevSearch/Car196_Combined/images/"

MODE = "search"  # "index" or "search"

# Load images from a folder
image_list = LoadData(data_type="image").from_folder([IMAGES_DIR])

print("Number of images: ", len(image_list))

# Initialize the custom feature extractor
feature_extractor = CustomFeatureExtractor(model_path=MODEL_PATH)

# Set up the search engine
search_engine = SearchSetup(
    item_list=image_list,
    feature_extractor=feature_extractor,
    dim_reduction=None,
    metadata_dir="Example/metadata_dir",
    feature_extractor_name="efficientnet_onnx",
    mode=MODE,
    item_loader=None,
)

# Example input image or path
input_image_path = "Example/car.jpg"

# Load input image
input_image = Image.open(input_image_path).resize((224, 224))

# Search by image
similar_images = search_engine.get_similar_items(
    item=input_image,
    number_of_items=10,
    return_paths=True,
)

print("Similar Images by inputing an image:", len(similar_images))

# Search by path
similar_images = search_engine.get_similar_items(
    item=input_image_path,
    number_of_items=10,
    return_paths=True,
)

print("Similar Images by inputing a path:", len(similar_images))

# Add new images to the index
search_engine.add_items_to_index(new_item_paths=[input_image_path])

# Search by image again after adding the new image
similar_images = search_engine.get_similar_items(
    item=input_image,
    number_of_items=10,
    return_paths=True,
)

print(
    "Similar Images by inputing an image after adding new image:", len(similar_images)
)
print(similar_images)
