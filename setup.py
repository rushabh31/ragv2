from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(requirements_path, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#') and not line.startswith('-e')]

# Read README.md safely
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Production-grade Retrieval-Augmented Generation (RAG) system"

setup(
    name="controlsgenai",
    version="0.1.0",
    description="Production-grade Retrieval-Augmented Generation (RAG) system",
    author="Windsurf",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    package_data={
        '': ['*.yaml', 'config/*.yaml'],
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
            'controlsgenai-ingestion=rag.ingestion.api.main:main',
            'controlsgenai-chatbot=rag.chatbot.api.main:main',
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
