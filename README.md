# ArxivLlama 🦙📚
> RAG-powered scientific research assistant for ArXiv papers

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

Transform how you interact with scientific literature using AI-powered analysis of ArXiv papers.

## Features
- 🔍 Intelligent paper discovery from ArXiv
- 🧠 RAG implementation with LlamaIndex
- 💬 Natural language querying of research papers
- 📊 Cost-efficient processing pipeline
- 📑 Source citation with relevance scores
- 💰 Cost tracking for OpenAI usage

## Tech Stack
```mermaid
graph LR
A[ArXiv API] --> B[PDF Processing]
B --> C[LlamaIndex]
C --> D[ChromaDB]
D --> E[GPT-4o]