from setuptools import setup, find_packages

import clip2frame


setup(name='clip2frame',
      version=clip2frame.__version__,
      description='Codes for event localization in music auto-tagging',
      url='https://github.com/ciaua/clip2frame',
      author='Jen-Yu Liu',
      author_email='ciaua@citi.sinica.edu.tw',
      license='ISC',
      packages=find_packages(),
      )
