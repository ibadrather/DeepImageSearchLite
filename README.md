
# DeepSearchLite
DeepSearchLite is a lightweight and versatile package for finding similar items in a large collection of data, initially inspired by [DeepImageSearch](https://github.com/TechyNilesh/DeepImageSearch). It supports a variety of data types, such as images, text, or any other type of data that can be represented by feature vectors. All you need is a suitable encoder to convert the data into a feature vector, and DeepSearchLite will handle the rest, enabling you to perform similarity searches efficiently and effectively.

## Features
* Supports various data types (images, text, etc.) as long as they can be represented by feature vectors
* Easy integration with custom feature encoders
* Fast similarity search using FAISS indexing
* Flexible and extensible with support for custom feature extraction, dimensionality reduction functions, and item loading functions
* Add new items to the index without the need for re-indexing the entire dataset
* Retrieve item metadata and perform search by providing either the item itself or a file path
 
## Installation
Install DeepSearchLite using pip:
```bash
    pip install DeepSearchLite

```
## Quickstart

To use DeepSearchLite, you need to provide a custom feature encoder that can convert your data into feature vectors. For example, let's say you have a collection of images, and you want to find similar images. You can use a pre-trained deep learning model as a feature encoder. Please refer to the [Example](./Example) folder for a complete example of setting up DeepSearchLite for image similarity search.
For other types of data, you need to provide a suitable encoder that can convert the data into feature vectors and a compatible item loader function if necessary.

## Customization
DeepSearchLite offers flexibility by allowing you to customize feature extraction, dimensionality reduction, and item loading functions to suit your specific use case. To incorporate your custom functions, provide them as arguments when initializing the `SearchSetup` instance:

```python
search_setup = SearchSetup(
    item_list=item_list,
    feature_extractor=custom_feature_extractor,
    dim_reduction=custom_dim_reduction, # Set to None if not required
    metadata_dir="metadata_dir",
    feature_extractor_name="custom_extractor_name",
    mode="search", # Choose between "index" or "search"
    item_loader=custom_item_loader
)

```
In the example above:
* `custom_feature_extractor` is a function that extracts features from the given items according to your requirements.
* `custom_dim_reduction` is an optional function that reduces the dimensionality of the feature vectors. Set it to `None` if you don't need dimensionality reduction.
* `custom_item_loader` is a function that loads items from their file paths, ensuring compatibility with the feature extractor.

By providing your custom functions, you can tailor DeepSearchLite to work with various data types and feature extraction techniques, enabling a more personalized and efficient similarity search experience.

## Usage
You can use the provided methods in the `SearchSetup` class to index and search for similar items, add new items to the index, retrieve item metadata, and more. Check the class documentation for a detailed list of available methods and their usage.

## MODE

There are two main modes available in the SearchSetup class: 'index' and 'search'. These modes are specified when initializing the `SearchSetup` class.

* #### Index Mode
    In the index mode, the metadata is initialized, which contains the item paths along with the feature vectors of these items. This mode should be used when you are initializing your `SearchSetup` for the first time. During the indexing process, a directory is created to store this metadata, which will be used later for similarity search.

    When using the index mode, the feature extractor, dimensionality reducer (if any), and item loader are applied to extract features from the items in the dataset. The resulting feature vectors are then saved in the metadata along with the item paths, and an index is created to enable fast similarity search.

* #### Search Mode

    In the search mode, you can search for similar items by providing the item itself or the item path as input. When initializing the `SearchSetup` class in search mode, make sure to use the same feature extractor, dimensionality reducer (if any), and item loader as when the index was created. This is essential to ensure that the features extracted during the search are compatible with the indexed features.

    During the search process, the specified item is loaded, and its features are extracted using the provided feature extractor and dimensionality reducer (if any). The search is then performed using the pre-built index, and a list of similar items from the indexed dataset is returned.

    In summary, the index mode is used to build the metadata and index required for similarity search, while the search mode enables you to search for similar items in the indexed dataset. Make sure to use the same feature extractor, dimensionality reducer, and item loader in both modes to ensure compatibility and accurate search results.




## Contributing
Contributions to DeepSearchLite are welcome! If you have an idea, bug report, or feature request, please open an issue on the [GitHub repository](https://github.com/ibadrather/DeepSearchLite). If you'd like to contribute code, please fork the repository and submit a pull request.

## License
DeepSearchLite is released under the [MIT License](https://github.com/yourusername/DeepSearchLite/blob/main/LICENSE).

------------------
