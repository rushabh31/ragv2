from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(requirements_path, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#') and not line.startswith('-e')]

setup(
    name="controlsgenai",
    version="0.1.0",
    description="Production-grade Retrieval-Augmented Generation (RAG) system",
    author="Windsurf",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=find_packages() + find_packages(where='src'),
    package_dir={'': '.', 'rag': 'src/rag'},
    include_package_data=True,
    package_data={
        'controlsgenai': ['config/*.yaml', 'config.yaml', 'config_sample.yaml'],
    },
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.4.2',
            'pytest-asyncio>=0.21.1',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.5.0',
        ],
        'test': [
            'pytest>=7.4.2',
            'pytest-asyncio>=0.21.1',
            'httpx>=0.24.1',
        ]
    },
    entry_points={
        'console_scripts': [
            'controlsgenai-ingestion=src.rag.src.ingestion.api.main:main',
            'controlsgenai-chatbot=src.rag.src.chatbot.api.main:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
