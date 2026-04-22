from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate # Import ChatPromptTemplate and  PromptTemplate from langchain_core.prompts
#from langchain_openai import ChatOpenAI, OpenAI # Import OpenAI from langchain_openai
from langchain_ollama import ChatOllama # Import ChatOllama from langchain_ollama


load_dotenv()  # Load environment variables from .env file
import os


def main():
    print("Hello from langchain!")
    Information = """Narendra Damodardas Modi[a] (born 17 September 1950) is an Indian politician who has served as the prime minister of India since 2014. Modi was the chief minister of Gujarat from 2001 to 2014 and is the member of parliament (MP) for Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a right-wing Hindutva paramilitary volunteer organisation. He is the longest-serving prime minister outside the Indian National Congress.

Modi was born and raised in Vadnagar, Bombay State (present-day Gujarat), where he completed his secondary education. He was introduced to the RSS at the age of eight, becoming a full-time worker for the organisation in Gujarat in 1971. The RSS assigned him to the BJP in 1985, and he rose through the party hierarchy, becoming general secretary in 1998.[b] In 2001, Modi was appointed chief minister of Gujarat and elected to the legislative assembly soon after. His administration is considered complicit in the 2002 Gujarat violence[c] and has been criticised for its management of the crisis. According to official records, a little over 1,000 people were killed, three-quarters of whom were Muslim; independent sources estimated 2,000 deaths, mostly Muslim, with many others raped, mutilated, or both.[4] A Special Investigation Team appointed by the Supreme Court of India in 2012 found no evidence to initiate prosecution proceedings against him, causing widespread anger and disbelief among the country's Muslim communities.[d] While his policies as chief minister were credited for encouraging economic growth, his administration was criticised for failing to significantly improve health, poverty and education indices in the state.[e]

In the 2014 Indian general election, Modi led the BJP to a parliamentary majority, the first for a party since 1984. His administration increased direct foreign investment and reduced spending on healthcare, education, and social-welfare programmes. Modi began a high-profile sanitation campaign and weakened or abolished environmental and labour laws. His demonetisation of banknotes in 2016 and introduction of the Goods and Services Tax in 2017 sparked controversy. Modi's administration launched the 2019 Balakot airstrike against an alleged terrorist training camp in Pakistan; the airstrike failed,[5][6] but the action had nationalist appeal.[7] Modi's party won the 2019 general election. In its second term, his administration revoked the special status of Jammu and Kashmir and introduced the Citizenship Amendment Act, prompting widespread protests and spurring the 2020 Delhi riots in which Muslims were targeted by Hindu mobs.[8][9][10] Three controversial farm laws led to sit-ins by farmers across the country, eventually causing their formal repeal. Modi oversaw India's response to the COVID-19 pandemic, during which, according to the World Health Organization, 4.7 million Indians died.[11][12] In the 2024 general election, Modi's party lost its majority in the lower house of Parliament and formed a government leading the National Democratic Alliance coalition. Following a terrorist attack in Indian-administered Jammu and Kashmir, Modi presided over the 2025 India–Pakistan conflict, which resulted in a ceasefire.

Under Modi's tenure, India has experienced democratic backsliding, or the weakening of democratic institutions, individual rights, and freedom of expression.[f][13][14] As prime minister, he has received consistently high approval ratings.[15][16][17] Modi has been described as engineering a political realignment towards right-wing politics. He remains a controversial figure domestically and internationally, over his Hindu nationalist beliefs and handling of the Gujarat riots, which have been cited as evidence of a majoritarian and exclusionary social agenda.[g]"""

    summary_template = """
Given the following information {Information} about a person , I want you to create:
1. A Short Summary
2. Two Interesting facts abiut them 
"""
    summary_prompt_template = PromptTemplate(
        input_variables=["Information"],
        template=summary_template
    )

    #llm = ChatOpenAI(model="gpt-5", temperature=0)  # Initialize the OpenAI language model with the specified parameters
    #temperature=0 ensures that the model generates deterministic responses, which is useful for tasks like summarization where consistency is important.
    
    llm = ChatOllama(model="gemma4:latest", temperature=0)  # Initialize the Ollama language model with the specified parameters
    
    chain = summary_prompt_template | llm  # Create a chain by combining the language model and the prompt template
    #first forms a prompt by filling in the template with the provided information, and then passes that prompt to the language model to generate a response.

    response = chain.invoke(input={"Information": Information})  # Generate a summary using the chain
    print(response.content)  # Print the generated summary



if __name__ == "__main__":
    main()
