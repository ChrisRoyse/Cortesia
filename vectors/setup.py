"""
Setup script for MCP RAG Indexer
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README
readme_path = Path(__file__).parent / "README_MCP.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="mcp-rag-indexer",
    version="1.0.0",
    description="MCP server for Universal RAG Indexing System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/mcp-rag-indexer",
    packages=find_packages(),
    py_modules=[
        'mcp_server',
        'indexer_universal',
        'query_universal',
        'git_tracker',
        'cache_manager'
    ],
    install_requires=[
        "mcp>=1.0.0",
        "langchain>=0.3.0",
        "langchain-community>=0.3.0",
        "langchain-huggingface>=0.1.0",
        "chromadb>=0.5.0",
        "sentence-transformers>=3.0.0",
        "gitpython>=3.1.0",
        "psutil>=5.9.0",
        "pyyaml>=6.0",
        "tomli>=2.0.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "mcp-rag-indexer=mcp_server:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)