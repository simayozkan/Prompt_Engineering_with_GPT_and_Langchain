!pip install openai==1.27
!pip install langchain==0.1.19
!pip install langchain-openai==0.1.6
!pip install langchain-experimental==0.0.58
!pip install typing_extensions==4.11.0
import os
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool


with open('financial_headlines.txt', 'r') as data:
    headlines = data.readlines()
headlines

headlines = [line.strip("\n") for line in headlines]
headlines

prompt_template = PromptTemplate.from_template(
    template="Analyze the following financial headline for sentiment: {headline}",
)

formatted_prompt = prompt_template.format(headline=headlines[0])

formatted_prompt


from langchain.prompts import ChatPromptTemplate

system_message = """You are performing sentiment analysis on news headlines regarding financial analysis. 
    This sentiment is to be used to advice financial analysts. 
    The format of the output has to be consistent. 
    The output is strictly limited to any of the following options: [positive, negative, neutral]."""

chat_template = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", "Analyze the following financial headline for sentiment: {headline}"),
])

formatted_chat_template = chat_template.format_messages(
    headline=headlines[0]
)

formatted_chat_template
from langchain.chains import LLMChain
client = OpenAI()

completion_chain = prompt_template | client

completion_chain.invoke({"headline": headlines[0]})

chat = ChatOpenAI()

chat_chain = chat_template | chat

chat_chain.invoke({"headline": headlines[0]}, {"system_message": system_message})

output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()

company_name_template = PromptTemplate(
    template="List all the company names from the following headlines, limited to one name per headline: {headlines}.\n{format_instructions}",
    input_variables=["headlines"],
    partial_variables={"format_instructions": format_instructions}
)

formatted_company_name_template = company_name_template.format(headlines=headlines)
formatted_company_name_template

model = OpenAI(temperature=0)

_output = model.invoke(formatted_company_name_template)

company_names = output_parser.parse(_output)

print(f"Data type: {type(company_names)}\n")

print(company_names)

from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool

agent_executor = create_python_agent(
    llm=OpenAI(temperature=0, max_tokens=1000),
    tool=PythonREPLTool(),
    verbose=True
)

agent_executor.invoke("What is the square root of 250? Round the answer down to 4 decimals.")

agent_executor.invoke(f"""For every of the following headlines, extract the company name and whether the financial sentiment is   positive, neutral or negative. 
   Load this data into a pandas dataframe. 
   The dataframe will have three columns: the name of the company, whether the financial sentiment is positive or negative and the headline itself. 
   The dataframe can then be saved in the current working directory under the name financial_analysis.csv.
   If a csv file already exists with the same name, it should be overwritten.

   The headlines are the following:
   {headlines}
   """)

import pandas as pd

df = pd.read_csv("financial_analysis.csv")

df

sentiment_template = PromptTemplate(
    template="Get the financial sentiment of each of the following headlines. The output is strictly limited to any of the following options: ['Positive', 'Negative', 'Neutral']: {headlines}.\n{format_instructions}",
    input_variables=["headlines"],
    partial_variables={"format_instructions": format_instructions}
)

formatted_sentiment_template = sentiment_template.format(headlines=headlines)

_output = model.invoke(formatted_sentiment_template)

sentiments = output_parser.parse(_output)
sentiments

def visualize_sentiments(headlines, sentiments):
    assert len(headlines) == len(sentiments)

    for i, _ in enumerate(headlines):
        print(f"{sentiments[i].upper()}: {headlines[i]}")

visualize_sentiments(headlines, sentiments)


#few shot examples 
sentiment_examples = """
    If a company is doing financially better than before, the sentiment is positive. For example, when profits or revenue have increased since the last quarter or year, exceeding expectations, a contract is awarded or an acquisition is announced.
    If the company's profits are decreasing, losses are mounting up or overall performance is not meeting expectations, the sentiment is negative.
    If nothing positive or negative is mentioned from a financial perspective, the sentiment is neutral.
"""

sentiment_template = PromptTemplate(
    template="Get the financial sentiment of each of the following headlines. {few_shot_examples} The output is strictly limited to any of the following options: ['Positive', 'Negative', 'Neutral']: {headlines}.\n{format_instructions}",
    input_variables=["headlines", "few_shot_examples"],
    partial_variables={"format_instructions": format_instructions}
)

formatted_sentiment_template = sentiment_template.format(headlines=headlines, few_shot_examples=sentiment_examples)

_output = model.invoke(formatted_sentiment_template)

sentiments = output_parser.parse(_output)

visualize_sentiments(headlines, sentiments)


agent_executor.invoke(f"""Create a dataframe with two columns: company_name, sentiment and headline.
To fill the dataframe, use the following lists respectively: {str(company_names)}, {str(sentiments)} and {str(headlines)}. 
The dataframe can then be saved in the current working directory under the name financial_analysis_with_parsing.csv.
If a csv file already exists with the same name, it should be overwritten.
""")

df = pd.read_csv("financial_analysis_with_parsing.csv")
df












