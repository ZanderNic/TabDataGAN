from setuptools import setup, find_packages

setup(
    name="table_gan",                         
    version="0.1.0",                           
    packages=find_packages(where="src"),     
    package_dir={"": "src"},                   
    install_requires=[                         
        "torch>=1.9.0",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
    ],
    author="Nicolas Zander",
    url="https://github.com/username/my_project", 
    classifiers=[                             
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",                   
)
