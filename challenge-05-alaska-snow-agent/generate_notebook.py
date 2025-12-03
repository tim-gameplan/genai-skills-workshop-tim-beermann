#!/usr/bin/env python3
"""
Generate complete Challenge 5 Jupyter Notebook from COMPLETE_IMPLEMENTATION_GUIDE.md

This script extracts all 11 code cells from the implementation guide and creates
a properly structured .ipynb file ready for execution in Google Colab.
"""

import json
import re
from pathlib import Path


def create_notebook_base():
    """Create the base notebook structure with metadata."""
    return {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            },
            "colab": {
                "provenance": [],
                "toc_visible": True,
                "name": "Challenge 5: Alaska Snow Agent"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }


def markdown_cell(text):
    """Create a markdown cell from text."""
    lines = text.split("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines]
    }


def code_cell(code):
    """Create a code cell from code string."""
    lines = code.split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines]
    }


def extract_cell_code(content, cell_number):
    """Extract code for a specific cell from markdown."""
    # Different patterns for different cells
    patterns = {
        1: r'### Cell 1: Environment Setup.*?```python\n(.*?)\n```',
        2: r'### Cell 2: Data Ingestion.*?```python\n(.*?)\n```',
        3: r'### Cell 3: Build Vector Search.*?```python\n(.*?)\n```',
        4: r'### Cell 4: Implement AlaskaSnowAgent.*?```python\n(.*?)\n```',
        5: r'### Cell 5: Create Model Armor.*?```python\n(.*?)\n```',
        6: r'### Cell 6: Enhanced Logging.*?```python\n(.*?)\n```',
        7: r'### Cell 7: Create pytest.*?```python\n(.*?)\n```',
        8: r'### Cell 8: LLM Evaluation.*?```python\n(.*?)\n```',
        9: r'### Cell 9: Generate Streamlit.*?```python\n(.*?)\n```',
        10: r'### Cell 10.*?```python\n(.*?)\n```',
        11: r'### Cell 11: Create Comprehensive.*?```python\n(.*?)\n```',
    }

    pattern = patterns.get(cell_number)
    if not pattern:
        return None

    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1)
    return None


def main():
    print("🚀 Generating Challenge 5 Jupyter Notebook")
    print("=" * 70)
    print()

    # Read the implementation guide
    guide_path = Path("COMPLETE_IMPLEMENTATION_GUIDE.md")
    if not guide_path.exists():
        print("❌ ERROR: COMPLETE_IMPLEMENTATION_GUIDE.md not found")
        return

    with open(guide_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Create notebook
    notebook = create_notebook_base()

    # Add title cell
    print("📝 Adding title cell...")
    title_md = """# Challenge 5: Alaska Department of Snow - Virtual Assistant

**Production-Grade RAG Agent for Snow Removal Information**

> Built for Public Sector GenAI Delivery Excellence Skills Validation Workshop

**Target Score:** 39-40/40 points (97-100%)

---

## 🎯 What You're Building

A production-quality AI chatbot that:
- Answers citizen questions about plowing schedules and school closures
- Uses RAG (Retrieval-Augmented Generation) with BigQuery vector search
- Integrates external APIs (Google Geocoding + National Weather Service)
- Implements comprehensive security (Model Armor)
- Includes automated testing (21+ pytest tests)
- Deploys to a public website (Streamlit on Cloud Run)

---

## 📋 Requirements Coverage

| # | Requirement | Implementation |
|---|-------------|----------------|
| 1 | Backend data store for RAG | BigQuery vector search |
| 2 | Access to backend API functionality | Geocoding + Weather APIs |
| 3 | Unit tests for agent functionality | 21+ pytest tests |
| 4 | Evaluation using Google Evaluation service | Vertex AI EvalTask |
| 5 | Prompt filtering and response validation | Model Armor |
| 6 | Log all prompts and responses | BigQuery logging |
| 7 | Generative AI agent deployed to website | Streamlit on Cloud Run |

---

## ⚡ Quick Start

1. Update `PROJECT_ID` in Cell 1
2. Run all cells sequentially
3. Wait for each cell to complete before proceeding
4. Monitor output for errors
5. Test agent with sample queries

---"""

    notebook["cells"].append(markdown_cell(title_md))

    # Extract and add all 11 cells
    cell_names = [
        "Cell 1: Environment Setup & Permissions",
        "Cell 2: Data Ingestion with Dynamic Discovery",
        "Cell 3: Build Vector Search Index (RAG Foundation)",
        "Cell 4: AlaskaSnowAgent Class (Core RAG Engine)",
        "Cell 5: Model Armor Security Template",
        "Cell 6: Enhanced Logging to BigQuery",
        "Cell 7: pytest Test Suite (21+ Tests)",
        "Cell 8: LLM Evaluation with Multiple Metrics",
        "Cell 9: Streamlit Web Application",
        "Cell 10: Architecture Diagram Generation",
        "Cell 11: Comprehensive README Documentation"
    ]

    for i in range(1, 12):
        print(f"📦 Extracting Cell {i}: {cell_names[i-1]}...", end=" ")

        code = extract_cell_code(content, i)

        if code:
            # Add markdown header
            notebook["cells"].append(markdown_cell(f"## {cell_names[i-1]}"))

            # Add code cell
            notebook["cells"].append(code_cell(code))
            print("✅")
        else:
            print("⚠️  NOT FOUND")

    # Write notebook
    output_path = Path("alaska_snow_agent_complete.ipynb")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print(f"✅ Notebook created: {output_path}")
    print(f"📊 Total cells: {len(notebook['cells'])}")

    # Count cell types
    markdown_count = sum(1 for c in notebook['cells'] if c['cell_type'] == 'markdown')
    code_count = sum(1 for c in notebook['cells'] if c['cell_type'] == 'code')

    print(f"   - Markdown cells: {markdown_count}")
    print(f"   - Code cells: {code_count}")
    print()
    print(f"📝 Next Steps:")
    print(f"   1. Open in Google Colab or Jupyter Lab")
    print(f"   2. Update PROJECT_ID in Cell 1")
    print(f"   3. Run all cells sequentially (Runtime → Run all)")
    print(f"   4. Monitor output and fix any errors")
    print(f"   5. Test the agent with sample queries")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
