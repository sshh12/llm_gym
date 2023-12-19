from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()


setup(
    name="llm_gym",
    version="0.0.1",
    description="",
    url="https://github.com/sshh12/llm_gym",
    author="Shrivu Shankar",
    license="Apache License 2.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
)
