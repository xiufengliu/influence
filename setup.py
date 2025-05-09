from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="dynamic-influence-clustering",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Dynamic Influence-Based Clustering Framework for Energy Consumption Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-github-username/dynamic-influence-clustering",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
)
