from setuptools import find_packages, setup

requirements = [
    'datasets==1.12.0',
    'soundfile',
    'librosa',
    'jiwer==2.2.1',
    'matplotlib',
    'torchaudio',
    'fairseq @ git+https://github.com/pytorch/fairseq.git@f3b6f5817fbee59057ae2506f01502ea3c301b4b',
]

setup(
    name='wav2vec',
    version='v0.1.0',
    packages=find_packages(),
    install_requires=requirements,
)
