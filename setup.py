from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="DeepSearchLite",
    version="0.2.1",
    author="Ibad Rather",
    author_email="ibad.rather.ir@gmail.com",
    description="A package for similarity search for any type of data. Use your own feature extractor and data loader and use this package to perform similarity search.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ibadrather/DeepSearchLite",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "Pillow",
        "faiss-cpu",
        "tqdm",
    ],
)
