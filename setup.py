# setup.py
from setuptools import setup, find_packages

setup(
    name="greentext_search",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "faiss-cpu",
        "Pillow",
        "gradio",
        "tqdm"
    ],
    description="Greentext CLIP + FAISS search package",
)

