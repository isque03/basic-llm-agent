from tavily import TavilyClient
import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

"""
This module contains tools for agentic LLM interactions, including web search capabilities using the Tavily API.
It provides functions to check for environment variables and perform web searches to retrieve recent information relevant to the user's queries.
"""


def check_environment_variable(var_name):
    """
    Check if the specified environment variable is set.

    Args:
        var_name (str): The name of the environment variable to check.

    Returns:
        bool: True if the environment variable is set, False otherwise.
    """
    return var_name in os.environ and os.environ[var_name] != ""


def extract_content(url):
    """
    Extracts content from a given URL using the Tavily API.

    Args:
        url (str): The URL from which to extract content.

    Returns:
        dict: A dictionary containing the extracted content with the following structure:
            - 'title': The title of the page.
            - 'content': The main content of the page.
            - 'images': List of image URLs found on the page.
            - 'links': List of links found on the page.
            - 'raw_content': Raw content data (may be None).
    """
    logger.info("Extracting content from URL: %s", url)
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    result = client.extract(url)
    if not result:
        raise ValueError(f"Failed to extract content from URL: {url}")
    return result


def web_search(query):
    """
    Performs a web search using the Tavily API and returns the results.

    This function initializes the Tavily client with the API key from environment variables
    and performs a search query using their search engine. Tavily specializes in providing
    up-to-date information from the web, making it suitable for retrieving recent news or trends.

    Args:
        query (str): The search query to perform.

    Returns:
        dict: A dictionary containing the search results with the following structure:
            - 'query': The original search query.
            - 'follow_up_questions': Suggested follow-up questions (may be None).
            - 'answer': A direct answer to the query if available (may be None).
            - 'images': List of relevant images if any.
            - 'results': A list of dictionaries, each containing:
                - 'title': The title of the search result.
                - 'url': The URL to the full content.
                - 'content': A text summary or snippet of the content.
                - 'score': A relevance score of the search result (float between 0-1).
                - 'raw_content': Raw content data (may be None).
            - 'response_time': Time taken to process the query in seconds.

    Example:
    ```
    results = web_search("latest NVIDIA stock price")
    print(f"Query: {results['query']}")
    print(f"Response time: {results['response_time']} seconds")
    for result in results['results']:
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Score: {result['score']}")
        print("---")
    ```

    Note:
        Ensure that the environment variable `TAVILY_API_KEY` is set with your Tavily API key before calling this function.
    """
    logger.info("Performing web search with query: %s", query)
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search(query=query)
    return results


def is_safe_math_expression(expression):
    """
    Checks if the provided expression is a safe mathematical expression.

    Args:
        expression (str): The mathematical expression to check.

    Returns:
        bool: True if the expression is safe, False otherwise.
    """
    # A simple check for safe characters (digits, operators, parentheses)
    allowed_chars = set("0123456789+-*/(). ")
    return all(char in allowed_chars for char in expression)


def calculator(expression):
    """
    Evaluates a mathematical expression and returns the result.

    WARNING: This function uses `eval()` to evaluate the expression, which can be dangerous if the input is not controlled. This should only be used with trusted input or in a controlled environment.

    Args:
        expression (str): The mathematical expression to evaluate.

    Returns:
        float: The result of the evaluated expression.

    Raises:
        ValueError: If the expression is unsafe or if there is an error during evaluation.
    Example:
        >>> result = calculator("3 + 5 * (2 - 8)")
        >>> print(result)
        -13.0
    Note:
        Ensure that the expression is safe before calling this function to avoid security risks associated with `eval()`.
    """
    try:
        if not is_safe_math_expression(expression):
            raise ValueError(
                "Unsafe mathematical expression detected. Please use only digits, operators, and parentheses."
            )
        # Evaluate the expression safely
        result = eval(expression)
        return result
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{expression}': {e}")


def market_data(symbol, data_type="price", period="1d", interval="30m"):
    """
    Fetches market data for trading analysis and decision-making.

    This function provides various types of financial market data including current prices,
    historical data, company information, and news for a given stock symbol using the yfinance API.

    Args:
        symbol (str): One stock ticker symbol (e.g., "AAPL", "MSFT", "NVDA")
        data_type (str): Type of data to retrieve:
            - "price": Current price and daily trading info (default)
            - "historical": Historical OHLCV (Open, High, Low, Close, Volume) data
            - "info": Company information and key financial statistics
            - "news": Recent news articles about the company
        period (str): Time period for historical data. Options include:
            - "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"
        interval (str): Data interval for historical data. Options include:
            - "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
            Note: Intraday data cannot extend last 60 days for "1m" interval

    Returns:
        dict: A dictionary containing the requested market data with the following structure:
            - 'symbol': The ticker symbol requested
            - 'timestamp': When the data was fetched (YYYY-MM-DD HH:MM:SS)
            - 'data_type': The type of data requested
            - 'data': The actual market data, structure depends on data_type:
                - For "price": {
                    "current_price": float,
                    "open": float,
                    "high": float,
                    "low": float,
                    "volume": int,
                    "change_percent": float
                }
                - For "historical": List of dictionaries, each containing:
                    {
                        "date": "YYYY-MM-DD HH:MM:SS",
                        "open": float,
                        "high": float,
                        "low": float,
                        "close": float,
                        "volume": int
                    }
                - For "info": {
                    "name": str,
                    "sector": str,
                    "industry": str,
                    "market_cap": int,
                    "pe_ratio": float,
                    "dividend_yield": float,
                    "52w_high": float,
                    "52w_low": float,
                    "avg_volume": int,
                    "description": str
                }
                - For "news": List of dictionaries, each containing:
                    {
                        "title": str,
                        "summary": str,
                        "content_type": str,
                        "publisher": str,
                        "published": str,
                        "link": str,
                        "thumbnail_url": str,
                        "editors_pick": bool,
                        "id": str
                    }

    Example:
        ```
        # Get current price
        price_data = market_data("AAPL")
        print(f"Current price of {price_data['symbol']}: ${price_data['data']['current_price']}")

        # Get historical data
        hist_data = market_data("MSFT", data_type="historical", period="5d")
        print(f"Last 5 closing prices: {[day['close'] for day in hist_data['data'][:5]]}")

        # Get news about a company
        news_data = market_data("NVDA", data_type="news")
        for article in news_data["data"]:
            print(f"{article['title']} - {article['publisher']}")
        ```

    Note:
        This function requires the yfinance package to be installed.
        Some data types and frequencies may be limited by the provider's API restrictions.
    """
    print(
        f"Fetching market data for {symbol} with data_type={data_type}, period={period}, interval={interval}"
    )
    try:
        ticker = yf.Ticker(symbol)
        result = {
            "symbol": symbol,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_type": data_type,
            "data": {},
        }

        if data_type == "price":
            # Get current price and basic info
            hist = ticker.history(period="3d")
            if hist.empty:
                raise ValueError(f"No data found for symbol: {symbol}")

            # Get the last available price data
            last_quote = hist.iloc[-1]
            result["data"] = {
                "current_price": round(last_quote["Close"], 2),
                "open": round(last_quote["Open"], 2),
                "high": round(last_quote["High"], 2),
                "low": round(last_quote["Low"], 2),
                "volume": int(last_quote["Volume"]),
                "change_percent": (
                    round((last_quote["Close"] / hist.iloc[-2]["Close"] - 1) * 100, 2)
                    if len(hist) > 1
                    else 0
                ),
            }

        elif data_type == "historical":
            # Get historical OHLCV data
            hist = ticker.history(period=period, interval=interval)
            if hist.empty:
                raise ValueError(
                    f"No historical data found for {symbol} with period={period}, interval={interval}"
                )

            # Convert historical data to a list of dictionaries
            hist_data = []
            for date, row in hist.iterrows():
                hist_data.append(
                    {
                        "date": date.strftime("%Y-%m-%d %H:%M:%S"),
                        "open": round(row["Open"], 2),
                        "high": round(row["High"], 2),
                        "low": round(row["Low"], 2),
                        "close": round(row["Close"], 2),
                        "volume": int(row["Volume"]),
                    }
                )
            result["data"] = hist_data

        elif data_type == "info":
            # Get company information
            info = ticker.info
            # Select the most relevant information
            result["data"] = {
                "name": info.get("shortName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "dividend_yield": (
                    info.get("dividendYield", 0) * 100
                    if info.get("dividendYield")
                    else 0
                ),
                "52w_high": info.get("fiftyTwoWeekHigh", 0),
                "52w_low": info.get("fiftyTwoWeekLow", 0),
                "avg_volume": info.get("averageVolume", 0),
                "description": info.get("longBusinessSummary", ""),
            }

        elif data_type == "news":
            # Get news about the company
            news_items = ticker.news

            result["data"] = []
            for item in news_items[:5]:  # Limit to 5 news items
                try:
                    news_entry = {
                        # Basic info
                        "title": item.get("content", {}).get("title", "No Title"),
                        "summary": item.get("content", {}).get("summary", ""),
                        "content_type": item.get("content", {}).get(
                            "contentType", "STORY"
                        ),
                        # Publication info
                        "publisher": item.get("content", {})
                        .get("provider", {})
                        .get("displayName", "Unknown Publisher"),
                        "published": item.get("content", {}).get("pubDate", ""),
                        # Links
                        "link": item.get("content", {})
                        .get("clickThroughUrl", {})
                        .get(
                            "url",
                            item.get("content", {})
                            .get("canonicalUrl", {})
                            .get("url", ""),
                        ),
                        # Image (if available)
                        "thumbnail_url": item.get("content", {})
                        .get("thumbnail", {})
                        .get("resolutions", [{}])[0]
                        .get("url", ""),
                        # Metadata
                        "editors_pick": item.get("content", {})
                        .get("metadata", {})
                        .get("editorsPick", False),
                        "id": item.get("id", ""),
                    }
                    result["data"].append(news_entry)
                except Exception as e:
                    logger.warning(f"Error processing news item: {e}")

            if not result["data"]:
                logger.warning(f"No news items could be processed for {symbol}")

        else:
            raise ValueError(
                f"Invalid data_type: {data_type}. Must be 'price', 'historical', 'info', or 'news'"
            )

        print(result)

        return result

    except Exception as e:
        raise ValueError(f"Error fetching market data for {symbol}: {str(e)}")


if __name__ == "__main__":
    # Example usage of the web_search function
    # Test market_data function
    try:
        # Get current price
        data = market_data("NVDA")
        print(data)
    except Exception as e:
        print(f"Error: {e}")
