import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tfencoder",
    version="0.0.1",
    author="Cheng Guo",
    author_email="guocheng672@gmail.com",
    description="A pytorch implementation of transformer encoder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/guocheng2018/transformer-encoder",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
