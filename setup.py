from setuptools import setup, find_packages

setup(
    name="delphi",
    version="0.1.0",
    author="Sanger Steel",
    author_email="sangersteel@gmail.com",
    description="A basic inference engine and server for GPT-NeoX models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sangstar/delphi",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
