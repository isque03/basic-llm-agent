# basic-llm-agent
A simple example of building an LLM agent from scratch in python.

Inspired by Aaron Dunn's great tutorial:
https://youtu.be/mYo7UFwnW1k?si=uSZ_fhA8rJPsApkz

## Overview
This project illustrates the construction of a simple Large Language Model (LLM) agent using Python. It serves as an educational example for developers who want to understand the fundamental concepts and implementation details of an LLM agent without relying on external agent frameworks.

## Features
- Built from scratch in Python with minimal dependencies
- Demonstrates the core concepts of LLM-based agents
- Includes various tools like web search, calculator, and market data analysis
- Web search capabilities through Tavily API

## Requirements
- Python 3.8+
- OpenAI API key (or other LLM provider)
- Tavily API key for web search functionality
- Required Python packages:
  - openai>=1.0.0
  - tavily-python>=0.5.1
  - python-dotenv>=1.0.0
  - yfinance>=0.2.54
  - pandas>=2.0.3

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/basic-llm-agent.git
   cd basic-llm-agent
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API keys:
   
   Create a `.env` file in the root directory with the following content:
   ```
   OPENAI_API_KEY=your_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

   Note: The `.env` file is included in the `.gitignore` file to ensure your API keys are never accidentally committed to version control.

   Alternatively, you can set these as environment variables:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   export TAVILY_API_KEY=your_tavily_api_key_here
   ```

   You can obtain a Tavily API key by signing up at [https://tavily.com/](https://tavily.com/)
   
   You can obtain an OpenAI API key by signing up at [https://platform.openai.com/](https://platform.openai.com/)

## Usage

Use the agent_query function in llm.py:

```python
from llm import agent_query

# Ask the agent to perform a task
response = agent_query("What is the sum of today's high temperature in Paris and Boston?")
print(response)
```

The repository includes both simple and advanced usage examples at the bottom of llm.py.

## License
MIT License


