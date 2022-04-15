from setuptools import setup

setup(
   name='simple_textmining',
   version='0.2',
   author='Q35joih4334',
   author_email='Q35joih4334@gmail.com',
   packages=['simple_textmining'],
   url='https://github.com/Q35joih4334/simple_textmining',
   license='LICENSE.txt',
   description='Simple textmining tool',
   long_description=open('README.md').read(),
   install_requires=['spacy', 'textacy', 'matplotlib', 'tqdm', 'wordcloud', 'pandas', 'numpy', 'xlsxwriter'],
)
