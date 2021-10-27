from setuptools import setup, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("VERSION", "r") as fh:
    version = fh.read().strip()

setup(
    name='cltl.asr',
    version=version,
    package_dir={'': 'src'},
    packages=find_namespace_packages(include=['cltl.*', 'cltl_service.*'], where='src'),
    data_files=[('VERSION', ['VERSION'])],
    url="https://github.com/leolani/cltl-asr",
    license='MIT License',
    author='CLTL',
    author_email='t.baier@vu.nl',
    description='ASR for Leolani',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.7',
    install_requires=['numpy>=1.21.2'],
    extras_require={
        "impl": [
            "cffi>=1.14.6",
            "importlib_resources>=5.2.2",
            "jiwer>=2.2.0",
            "sounddevice>=0.4.2",
            "soundfile>=0.10.3.post1",
            "torch==1.9.0",
            "transformers==4.10.0"
        ],
        "service": [
            "cltl.backend",
            "cltl.combot",
            "cltl.vad",
            "emissor",
            "requests",
        ]
    },
)

