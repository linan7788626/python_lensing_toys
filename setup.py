"""
Usage:
    python setup.py py2app

"""

from setuptools import setup

APP = ['gg_show.py']
OPTIONS = {'iconfile':'222.icns',}

setup(
    app = APP,
    options = {'py2app': OPTIONS},
    setup_requires = ['py2app'],
)
