import os
import setuptools

################################################################
# Generate requirements list
requirementPath = './requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

################################################################
# ReadMe File
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MML",  # Replace with your own username
    version="0.0.1",
    author="Shubham Aggarwal",
    author_email="author@example.com",
    description="MML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Demigod2808/MML.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=install_requires,
)
