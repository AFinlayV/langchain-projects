"""
This is a script to use the OpenAI API to generate journal entries.
The general idea is to use GPT to generate questions to ask the user, then record the user's answers to those questions.
then it will save the answers to a file, and then generate more questions, and so on.
It can then generate daily and weekly summaries of the user's answers formatted as journal entries.
Then the entries can be embedded and semantically searchable in a database.
Ideally it can then use previous entries as context to generate new questions.

TODO:
    - [x] Set up OpenAI API
    - [x] Set up LLM
    - [x] make topic list
    - [x] generate questions
    - [x] Set up answers
    - [x] file saving
    - [x] daily summary
    - [ ] Rewrite to use a langchain agent and generate the questions dynamically, rather than all at once
    - [ ] Set up weekly summary
    - [ ] Implement whisper.ai to take in audio and convert to text
"""

from langchain.llms import OpenAI
from langchain import PromptTemplate
import os
import datetime

VERBOSE = False

TOPIC_LIST = {
    "Reflection": "questions about your day, thoughts, and feelings.",
    "Goals": "questions about the goals you want to achieve and how to reach them.",
    "Gratitude": "questions about what you are thankful for and why.",
    "Relationships": "questions about your relationships and how to strengthen them.",
}


def set_api_keys():
    with open('/Users/alexthe5th/Documents/API Keys/OpenAI_API_key.txt', 'r') as f:
        key = f.read().strip()
    os.environ['OPENAI_API_KEY'] = key


def init_llm(temperature=0.7):
    vprint("Loading LLM...")
    llm = OpenAI(temperature=temperature, model_name="text-davinci-003", verbose=VERBOSE, max_tokens=512)
    vprint("LLM loaded.", llm)
    return llm


def vprint(*args):
    if VERBOSE:
        print(*args)


def get_questions(llm, topic, description, num=3):
    """
    generates questions based on the topic and description given
    :return: list of questions
    """

    questions = []
    template = """
    I want you to a list of questions about {topic}.
    
    The description of the topic is: {description}
    
    You are going to generate the questions in the following format:
    1. How did you feel today?
    2. What did you do today?
    3. What did you learn today?
    
    Generate {num} questions.
    """

    prompt = PromptTemplate(
        input_variables=['topic', 'description', 'num'],
        template=template,
    )
    vprint(prompt)
    response = llm(prompt.format(num=num, topic=topic, description=description))
    for line in response.splitlines():
        if line:
            questions.append(line)
    vprint(questions)
    return questions


def generate_folowup_question(llm, question, answer, num=3):
    """
    generates follow-up questions based on the question and answer given
    :param num: number of questions to generate
    :param answer: user's answer
    :param llm: LLM to use
    :param question: question asked by llm
    :return: follow-up question
    """
    questions = []
    template = """
    I want you to generate a set of follow up question based on the question and answer given.
    Question:
    {question}
    Answer:
    {answer}

    You are going to generate the questions in the following format:
    1. How did you feel today?
    2. What did you do today?
    3. What did you learn today?

    Generate {num} questions.
    """
    prompt = PromptTemplate(
        input_variables=['num', 'question', 'answer'],
        template=template,
    )
    response = llm(prompt.format(num=num, question=question, answer=answer))
    for line in response.splitlines():
        if line:
            questions.append(line)
    return questions


def ask_question(question):
    """
    asks the questions and records the answers
    :param question: list of questions
    :return: dict of questions and answers
    """
    answer = input(f'Question: {question}:\n'
                   f'Answer: ')
    return answer


def generate_summary(llm, answer_dict):
    """
    generates a daily summary of the answers
    :param llm: The LLM to use
    :param answer_dict: dict of questions and answers
    :return: daily summary
    """
    template = """
    A User was asked a question and they gave the following response:
    Question:
    {question}
    Answer:
    {answer}
    Generate a section daily journal entry based on the responses.
    Do not include an introduction or conclusion, just the portion of the entry that would be a summary of the answer.
    Write the summary as if it is a continuation of a journal entry (i.e don't start with "Today I did ..."). 
    Use the first person and the present tense as if you are the user who has written the responses.
    Use interesting and descriptive language that will make the entry interesting to read,
    but don't add any extra details to the journal entry, only include details mentioned in the answer.
    Use variety in your sentence structure and vocabulary.
    Correct any spelling or grammar mistakes.
    """
    prompt = PromptTemplate(
        input_variables=['question', 'answer'],
        template=template,
    )
    summary = ""
    for key in answer_dict:
        summary += llm(prompt.format(question=key, answer=answer_dict[key]))
    return summary


def save_journal_entry(answer_dict, summary):
    """
    saves the journal entry to a file
    :param answer_dict: dict of questions and answers
    :param summary: daily summary
    :return: None
    """
    with open("journal.txt", "a") as f:
        f.write(f"Journal Entry for {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Questions and Answers:\n")
        for key in answer_dict:
            f.write(f"{key}: {answer_dict[key]}\n")
        f.write("Summary:\n")
        f.write(summary)


def main():
    set_api_keys()
    llm = init_llm()
    answer_dict = {}
    questions = []

    for topic in TOPIC_LIST:
        topic, description = topic, TOPIC_LIST[topic]
        questions += get_questions(llm, topic, TOPIC_LIST[topic])
        vprint(topic, ":", description)
    for question in questions:
        answer_dict[question] = ask_question(question)
        followups = generate_folowup_question(llm, question, answer_dict[question])
        for followup in followups:
            answer_dict[followup] = ask_question(followup)
    summary = generate_summary(llm, answer_dict)
    print(summary)
    save_journal_entry(answer_dict, summary)


if __name__ == '__main__':
    main()
