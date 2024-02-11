import os
import subprocess
from pathlib import Path
from typing import Union

from PIL import Image

from deepsearchlite.DeepSearchLite import SearchSetup
from deepsearchlite.utils import LoadData
from Example.FeatureExtractor import CustomFeatureExtractor

# Clear the screen in a cross-platform way
subprocess.run("cls" if os.name == "nt" else "clear", shell=True)

MODEL_PATH = Path("Example/feature_encoder.onnx")
IMAGES_DIR = Path("/home/ibad/Desktop/RevSearch/Car196_Combined/images/")

MODE = "search"  # "index" or "search"

# Load images from a folder
image_list = LoadData(data_type="image").from_folder([IMAGES_DIR])

print("Number of images: ", len(image_list))

# Initialize the custom feature extractor
feature_extractor = CustomFeatureExtractor(
    model_path=str(MODEL_PATH)
)  # Ensure the path is converted to string if needed

# Set up the search engine
search_engine = SearchSetup(
    item_list=image_list,
    feature_extractor=feature_extractor,
    dim_reduction=None,
    metadata_dir="Example/metadata_dir",
    feature_extractor_name="efficientnet_onnx",
    mode=MODE,
    item_loader=None,  # Define this if you're handling non-image data or need custom preprocessing
)

# Example input image or path
input_image_path = Path("Example/car.jpg")

# Load input image
input_image = Image.open(input_image_path).resize((224, 224))


# Define a function to search and print results
def search_and_print(
    search_engine: SearchSetup,
    item: Union[Image.Image, Path],
    number_of_items: int = 10,
    by_path: bool = False,
) -> None:
    if by_path:
        # Ensure item is a Path and convert to string. If item is already a Path, this is fine.
        # If item is not a Path (i.e., an Image), this branch wouldn't logically be executed.
        item_to_search = str(item)
    else:
        # If not by_path, the item should be an Image, and specific handling could be here.
        # For simplicity, this assumes you have to convert Image to a suitable format
        # (e.g., a file path or an image array).
        # This placeholder doesn't perform any conversion; actual implementation may vary.
        item_to_search = item

    similar_images = search_engine.get_similar_items(
        item=item_to_search,
        number_of_items=number_of_items,
        return_paths=True,
    )

    # Adjusted print statement to reflect the logic more accurately
    action = "by inputting a path" if by_path else "by inputting an image"
    print(f"Similar Images {action}: {len(similar_images)}")


# Perform searches
print("Similar Images by inputing an image:")
search_and_print(search_engine, input_image)

print("\nSimilar Images by inputing a path:")
search_and_print(search_engine, input_image_path, by_path=True)

# Add new images to the index
search_engine.add_items_to_index(new_item_paths=[str(input_image_path)])

# Perform search again after adding the new image
print("\nSimilar Images by inputing an image after adding new image:")
similar_images_after_adding = search_and_print(search_engine, input_image)
print(similar_images_after_adding)
