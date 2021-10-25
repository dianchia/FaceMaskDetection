from setuptools import find_packages, setup

setup(
    name="facemaskdetection",
    version="0.1.0",
    author="Chia Dian Rui",
    description="Face Mask Detector",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "numpy",
        "opencv-python",
        "mediapipe",
        "tensorflow",
        "rich",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
