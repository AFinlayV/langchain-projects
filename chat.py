from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import load_tools
from langchain.agents import initialize_agent
import os

VERBOSE = True


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

    with open('/Users/alexthe5th/Documents/API Keys/news-api-key.txt', 'r') as f:
        key = f.read().strip()
    os.environ['NEWS_API_KEY'] = key


def init_all():
    llm = init_llm()
    tools = init_tools(llm)
    memory = init_memory()
    agent = init_agent(tools, llm, memory)
    return agent


def init_llm(temperature=0.7):
    if VERBOSE:
        print("Loading LLM...")
    llm = OpenAI(temperature=temperature, model_name="text-davinci-003", verbose=VERBOSE, max_tokens=512)
    if VERBOSE:
        print("LLM loaded.", llm)
    return llm


def init_tools(llm):
    if VERBOSE:
        print("Loading tools...")
    tool_list = ["serpapi", "llm-math", 'wolfram-alpha', 'open-meteo-api', 'news-api']
    tools = load_tools(tool_list, llm=llm, news_api_key=os.environ['NEWS_API_KEY'])
    if VERBOSE:
        print("Tools loaded.")
        print("Tool list:", tool_list)
    return tools


def init_memory():
    if VERBOSE:
        print("Initializing memory...")
    memory = ConversationBufferMemory(memory_key="chat_history")
    if VERBOSE:
        print("Memory initialized.", memory)
    return memory


def init_agent(tools, llm, memory):
    if VERBOSE:
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


def conversation(agent, text):
    try:
        response = f'AI: {agent.run(input=text)}'
    except Exception as e:
        response = f'<Error>: {e}'
    return response


def summarize_chat_history(agent):
    if VERBOSE:
        print("Summarizing chat history...")
    if len(agent.memory.buffer) == 0:
        return 'No chat history found.'
    else:
        try:
            prompt = f'Please summarize the following text {agent.memory.buffer}.'
            response = agent.run(input=prompt)
        except Exception as e:
            response = f'<Error>: {e} \n no summary generated.'
    return response


def clear_chat_history(agent):
    prompt = f'The Human would like to clear the chat history.\nIs that ok with you?\n' \
             f'Please respond "Yes" if you consent to having your memory cleared.'
    response = agent.run(input=prompt)
    print(f'Getting consent to clear memory:\n{prompt}\n{response}')
    if 'Yes' in response or 'yes' in response:
        agent.memory.clear()
        save_chat_history(agent)
        response = 'AI memory cleared.'
    else:
        response = 'AI did not consent to clearing memory. Memory not cleared.'
    return response


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
        agent.memory.buffer = data['buffer']
    except FileNotFoundError:
        print("No chat history found. Starting new chat history.")
    except Exception as e:
        print(e)


def main():
    set_api_keys()
    agent = init_all()
    load_chat_history(agent)
    summary = summarize_chat_history(agent)
    print(f'Welcome to the chatbot! Type <clear> to clear the chat history.\nSummary of chat history:\n{summary}')
    while True:
        text = input('Human: ')
        if text:
            if text == '<clear>':
                print(clear_chat_history(agent))
            else:
                print(conversation(agent, text))
                save_chat_history(agent)
        else:
            break


if __name__ == '__main__':
    main()
