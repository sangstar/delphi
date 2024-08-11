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
        # List your project’s dependencies here
        "torch",
        "transformers",
    ],
    classifiers=[
        # See https://pypi.org/classifiers/ for a full list of available classifiers
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Specify Python version requirement
    entry_points={
        "console_scripts": [
            "your_command=your_module:main_function",
        ],
    },
)
