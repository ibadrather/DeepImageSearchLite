# DeepSearchLite
DeepSearchLite is a lightweight and versatile package for finding similar items in a large collection of data, adopted from [DeepImageSearch](https://github.com/TechyNilesh/DeepImageSearch). It supports a variety of data types such as images, text, or any other type of data that can be represented by feature vectors. All you need is a suitable encoder to convert the data into a feature vector, and DeepSearchLite will take care of the rest, enabling you to perform similarity searches efficiently and effectively.

## Features
* Supports various data types (images, text, etc.) as long as they can be represented by feature vectors
* Easy to use with custom feature encoders
* Fast similarity search using FAISS indexing
* Extensible with support for custom feature extraction and dimensionality reduction functions
* Add new items to the index without the need for re-indexing the entire dataset

## Installation
Install DeepSearchLite using pip:
```bash
    pip install DeepSearchLite
```
## Quickstart
To use DeepSearchLite, you need to provide a custom feature encoder that can convert your data into feature vectors.
For example, let's say you have a collection of images and you want to find similar images. You can use a pre-trained deep learning model as a feature encoder. Please refer to the [Example](./Example) folder for a complete example on how to set up DeepSearchLite for image similarity search.
For other types of data, you need to provide a suitable encoder that can convert the data into feature vectors.


## Customization
DeepSearchLite allows you to customize feature extraction and dimensionality reduction functions. You can provide your custom functions when initializing the `SearchSetup` instance:

```python
search_setup = SearchSetup(
    image_list=image_list,
    feature_extractor=custom_feature_extractor,
    dim_reduction=custom_dim_reduction_function
)
```
## Contributing
Contributions to DeepSearchLite are welcome! If you have an idea, bug report, or feature request, please open an issue on the [GitHub repository](https://github.com/yourusername/DeepSearchLite). If you'd like to contribute code, please fork the repository and submit a pull request.

## License
DeepSearchLite is released under the [MIT License](https://github.com/yourusername/DeepSearchLite/blob/main/LICENSE).

------------------