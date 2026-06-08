from setuptools import setup, find_packages

setup(
    name="codec-video-gen",
    version="0.1.0",
    author="Ishmael Affum Kwakye",
    description="Codec-inspired temporal design for CPU-native video generation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/IshCPU-VideoGenLab/codec-video-gen",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "einops>=0.7.0",
        "transformers>=4.35.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.7.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
    entry_points={
        "console_scripts": [
            "codec-video-gen=codec_video_gen.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
)
