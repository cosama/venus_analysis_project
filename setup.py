import setuptools

setuptools.setup(
    name="venus_analysis_utils",
    version="0.1.0",
    url="https://github.com/cosama/venus_analysis_project/",
    author="",
    author_email="",
    description="A collection of programs to parse and analyze PLC data.",
    packages=setuptools.find_packages(),
    install_requires=["pandas", "pyarrow"]
)
