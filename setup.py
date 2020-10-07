from setuptools import setup, find_packages

setup(
  name = 'performer-pytorch',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'Performer - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/performer-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'efficient attention',
    'transformers'
  ],
  install_requires=[
    'torch>=1.6',
    'einops>=0.3'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)