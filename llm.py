import re
from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
from tools import (
    check_environment_variable,
    web_search,
    calculator,
    market_data,
    extract_content,
)


class Agent:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Set up logging
        logging.basicConfig(level=logging.INFO)

        # Initialize OpenAI API with the API key from environment variables
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logging.error(
                "API key not found. Please set the OPENAI_API_KEY environment variable."
            )
            raise ValueError("API key is required.")

        self.client = OpenAI(api_key=self.api_key)
        # load prompt from system_prompt.txt
        with open("system_prompt.md", "r") as file:
            self.system_message = file.read().strip()
        self.messages = []
        self.messages.append({"role": "system", "content": self.system_message})

    def generate_response(self, prompt):
        """Generate a response from the LLM given a prompt."""
        try:
            self.messages.append({"role": "user", "content": str(prompt)})
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.messages,
                temperature=0.3,
            )
            self.messages.append(
                {"role": "assistant", "content": response.choices[0].message.content}
            )
            # Log the assistant's response
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return None


def extract_action(message):
    # Match either "Action: <action>" or "Action Input: <input>"
    action_regex = re.compile("^Action: (.+)$")
    action_input_regex = re.compile("^Action Input: (.+)$")

    lines = message.split("\n")
    action = None
    action_input = None
    for line in lines:
        action_match = action_regex.match(line)
        if action_match:
            action = action_match.group(1).strip()
            continue
        action_input_match = action_input_regex.match(line)
        if action_input_match:
            action_input = action_input_match.group(1).strip()
            continue
    return action, action_input


def parse_action_input_as_parameters(action_input):
    """Parse action input into args and kwargs."""
    if not action_input:
        return [], {}  # Return empty collections instead of None

    import shlex  # For proper shell-like argument parsing
    import ast
    import json

    # Handle JSON/dictionary-like inputs
    if action_input.strip().startswith("{") and action_input.strip().endswith("}"):
        try:
            # Parse as JSON
            parsed_dict = json.loads(action_input)

            # If it's a dictionary, convert to kwargs
            if isinstance(parsed_dict, dict):
                return [], parsed_dict
        except json.JSONDecodeError:
            # If JSON parsing fails, continue with normal parsing
            pass

    # Handle special case of simple quoted string (common for web_search)
    if (
        action_input.startswith('"')
        and action_input.endswith('"')
        and action_input.count('"') == 2
    ):
        return [action_input.strip('"')], {}

    # Handle quoted strings and proper tokenization
    try:
        tokens = shlex.split(action_input)
    except ValueError:
        # Fallback if shlex parsing fails
        tokens = action_input.split()

    args = []
    kwargs = {}

    for token in tokens:
        if "=" in token:
            key, value = token.split("=", 1)
            # Try to convert string values to Python types
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass  # Keep as string if conversion fails
            kwargs[key.strip()] = value
        else:
            args.append(token)

    return args, kwargs


def extract_answer(message):
    # Match "Final Answer: <answer>" or "Answer: <answer>"
    answer_regex = re.compile("^(?:Final Answer|Answer): (.+)$")
    lines = message.split("\n")
    for line in lines:
        answer_match = answer_regex.match(line)
        if answer_match:
            return answer_match.group(1).strip()
    return None


def agent_query(user_input, max_turns=20):
    """
    Process a user query through an AI agent with a limited number of interaction turns.
    This function creates an agent instance and handles a conversation loop, allowing
    the agent to generate responses, take actions, and eventually provide an answer.
    Parameters:
    -----------
    user_input : str
      The initial query or prompt from the user.
    max_turns : int, optional
      Maximum number of back-and-forth interactions allowed (default is 15).
    Returns:
    --------
    str or None
      The extracted answer from the agent's final response if successful,
      or None if max_turns is reached without a final answer or if response generation fails.
    Notes:
    ------
    The function relies on several external components:
    - Agent class with a generate_response method
    - extract_action and extract_answer functions to parse responses
    - known_actions dictionary mapping action names to callable functions
    The conversation continues until either:
    1. The agent provides a final answer (no further actions needed)
    2. Maximum number of turns is reached
    3. Response generation fails
    """

    agent = Agent()
    num_turns = 0
    original_input = user_input  # Store the original input

    # Define thresholds for warnings
    warning_threshold = max_turns * 0.6  # Start warning at 60% of max turns
    urgent_threshold = max_turns * 0.8  # Urgent warning at 80% of max turns

    while num_turns < max_turns:
        try:
            num_turns += 1

            # Add turn count warnings as we approach the limit
            current_input = user_input

            if num_turns > urgent_threshold:
                turn_warning = f"\n\nUrgent: You are at turn {num_turns} of {max_turns}. Please provide a final answer immediately as you are about to reach the maximum allowed turns."
                current_input = user_input + turn_warning
                logging.warning(f"Urgent warning: {turn_warning}")
            elif num_turns > warning_threshold:
                turn_warning = f"\n\nNote: You are at turn {num_turns} of {max_turns}. Please start wrapping up your analysis and work toward a final answer soon."
                current_input = user_input + turn_warning
                logging.info(f"Warning: {turn_warning}")

            # Log the turn count
            logging.info(f"Agent interaction - Turn {num_turns}/{max_turns}")

            response = agent.generate_response(current_input)
            print(f"Response from agent (Turn {num_turns}/{max_turns}):", response)

            if not response:
                logging.error("Failed to generate a response from the agent.")
                return None

            action, action_input = extract_action(response)
            answer = extract_answer(response)

            # Return answer if provided
            if answer:
                logging.info(f"Final answer provided after {num_turns} turns")
                return answer

            if action and action in known_actions:
                # Execute the action with the provided input
                args, kwargs = parse_action_input_as_parameters(action_input)
                try:
                    if args or kwargs:
                        result = known_actions[action](*args, **kwargs)
                    else:
                        result = known_actions[action]()
                    user_input = str(result)
                except Exception as e:
                    logging.error(f"Error executing action {action}: {e}")
                    user_input = f"Error executing {action}: {str(e)}"
            else:
                # If no action is specified but also no answer, remind to provide a final answer
                if not answer and num_turns > warning_threshold:
                    return "No final answer was provided. Based on the analysis so far, please provide your conclusion."
                return extract_answer(response)
        except Exception as e:
            logging.error(f"Error during agent interaction: {e}")
            user_input = f"We encountered an error: {str(e)}\n\nPlease check your tool arguments or continue solving your problem in another way."

    logging.warning(f"Max turns ({max_turns}) reached without a final answer.")
    return "Analysis exceeded maximum allowed turns. Based on the research completed, here is the current conclusion."


known_actions = {
    "web_search": web_search,
    "calculator": calculator,
    "market_data": market_data,
    "extract_content": extract_content,
}

if __name__ == "__main__":

    advanced_user_prompt = """Conduct a detailed analysis of potential stock investments as of March 2025, considering recent improvements in AI agents and thier impact on key econimic indicators. Focus on the following aspects:

    1. **Research Methodology:**
      - Utilize web search tools to identify best practices for conducting thorough company analysis.
      - Learn how to effectively evaluate company financial health, competitive positioning, and growth prospects.

    2. **Company Selection:**
      - Identify publicly traded yet unknown companies that will significantly benefit from recent advancements in AI technology.

    3. **Stock Price Trends:**
      - Collect and analyze stock price data over the past 30 days and 6 months for each selected company. Be sure you have the correct data for the time period.
      - Evaluate whether each company's stock demonstrates favorable trends for investment opportunities (e.g., upward momentum, recovery potential after decline, etc.).

    4. **Recent News and Events:**
      - Summarize recent relevant news articles or events affecting each selected company.
      - Evaluate how these events influence the company’s stock performance and investment potential.

    4. **Investment Recommendation:**
      - Determine if each stock is currently undervalued or overvalued based on the analysis.
      - Assess whether it is a favorable time to buy given current conditions and market sentiment.

    5. **Comprehensive Report:**
      - Present a detailed summary for each analyzed company, including:
        - Recent price trends
        - Impact AI advancements have on the company
        - Significant recent news
        - Final investment assessment and rationale

    5. **Web Research:**
      - Utilize credible financial news websites, market analysis platforms, and official company reports to inform your analysis.

    Provide a final summary highlighting the top recommended stocks for investment, including clear justifications based on your findings.
    """

    simple_user_prompt = (
        "What is the sum of today's high temperature in Paris and Boston?"
    )

    response = agent_query(simple_user_prompt)
    if response:
        print("Assistant:", response)
    else:
        print("Failed to generate a response.")
