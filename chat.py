from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import load_tools
from langchain.agents import initialize_agent
import os

VERBOSE = False


# Set API keys
def set_api_keys():
    with open('/Users/alexthe5th/Documents/API Keys/OpenAI_API_key.txt', 'r') as f:
        key = f.read().strip()
    os.environ['OPENAI_API_KEY'] = key

    with open('/Users/alexthe5th/Documents/API Keys/SerpAPI_key.txt', 'r') as f:
        key = f.read().strip()
    os.environ['SERPAPI_API_KEY'] = key

    with open('/Users/alexthe5th/Documents/API Keys/wolfram_id.txt', 'r') as f:
        key = f.read().strip()
    os.environ['WOLFRAM_ALPHA_APPID'] = key


def init_agent():
    if VERBOSE:
        print("Loading LLM...")
    llm = OpenAI(temperature=0.7)
    if VERBOSE:
        print("LLM loaded.", llm)
        print("Loading tools...")
    tool_list = ["serpapi", "llm-math", 'wolfram-alpha']
    tools = load_tools(tool_list, llm=llm)
    if VERBOSE:
        print("Tools loaded.")
        print("Tool list:", tool_list)
        print("Initializing memory...")
    memory = ConversationBufferMemory(memory_key="chat_history")
    if VERBOSE:
        print("Memory initialized.", memory)
        print("Initializing agent...")
    agent = initialize_agent(
        tools,
        llm,
        agent="conversational-react-description",
        verbose=VERBOSE,
        memory=memory,
    )
    if VERBOSE:
        print("Agent initialized.", agent)
    return agent


def conversation_loop(agent):
    while True:
        text = input("You: ")
        if text:
            print("Bot:", agent.run(input=text))
        else:
            break


def save_chat_history(agent, filename='chat_history.txt'):
    if VERBOSE:
        print(f"Saving chat history as {filename}...")
    with open(filename, 'w') as f:
        for line in agent.memory:
            f.write(str(line).strip() + '\n')
    if VERBOSE:
        print("Chat history saved.")


def load_chat_history(agent, filename='chat_history.txt'):
    if VERBOSE:
        print(f"Loading chat history from {filename}...")
    try:
        data = {}
        with open(filename, 'r') as f:
            for line in f.read().splitlines():
                if VERBOSE:
                    print(line)
                line = eval(line)
                key, value = line
                data[key] = value
        agent.memory = ConversationBufferMemory(memory_key="chat_history", buffer=data['buffer'])
    except FileNotFoundError:
        print("No chat history found.")


def main():
    set_api_keys()
    agent = init_agent()
    clear_chat_history = input("Clear chat history? (y/n): ")
    if clear_chat_history == 'y' or clear_chat_history == 'Y':
        agent.memory.clear()
    else:
        load_chat_history(agent)
    conversation_loop(agent)
    save_chat_history(agent)


if __name__ == '__main__':
    main()
