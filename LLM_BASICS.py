#What is Language Model?
############################
#A language model is a statistical model that predicts the likelihood of a sequence of words. For example, given a sequence of words, a language model can predict the next word in the sequence or assign a probability to a given sequence of words.
# The dog wigged its .Here, the language model would predict that the next word is likely to be "tail" or "head" rather than "car" or "house". Language models are used in various applications such as natural language processing, machine translation, and text generation.
# Language models can be trained on large corpora of text data to learn the patterns and structures of language, allowing them to generate coherent and contextually relevant text.
# #e can directly go ahead and use them through APIs to get our tasks done, like translating some text from one language to another language and analyzing sentiments and even generating new text out of it, which you can find over the internet. 
# That's completely a new data because these models are very capable of generating new text out of whatever data that they have gone through. 
# The most popular language models are based on the transformer architecture, which has revolutionized the field of natural language processing. Some examples of transformer-based language models include BERT, GPT-2, and GPT-3.


#What exactly is the framework?
####################################
# Suppose you came to my office and I give you a task of cutting a piece of paper into a dimension of 5 m by 5 m. 
# So what would you do? We'll go ahead and, and cut the piece of paper into five by 5 m length. 
# If on the other hand, I give you 1000 identical pieces of paper and ask you to do the same thing, cut that particular piece of paper into five by five in the 
# same way. No, you'll come up with a framework so that you can save your time. So that's what we call a framework. In this case, it may create a frame of five
#  by 5 m and use it for the rest of the pieces of papers to get the task done efficiently. 

# So now comes what is software framework? 
#########################################
# A framework is something that we have already discussed when it comes to software, it is nothing but something that provides the basic functionality which
#  we can use by extending it to create more complex applications. 
# It's simply a set of tools given to you which you can modify or add new components to it. 
# In the context of language models, a software framework provides the necessary tools and libraries to build, train, and deploy language models. 
# It abstracts away the complexities of implementing the underlying algorithms and allows developers to focus on building applications that utilize language models effectively. 
# Examples of software frameworks for language models include TensorFlow, PyTorch, and Hugging Face's Transformers library. 
# These frameworks provide pre-built components and functionalities that make it easier to work with language models and accelerate the development process.



#What is a Prompt?
#################
# A prompt is a piece of text that serves as an input to a language model, guiding it to generate a specific response.
# For example, if you want a language model to generate a story, you might provide a prompt like "Once upon a time, in a faraway land, there lived a brave knight named Sir Lancelot." The language model would then generate a continuation of the story based on the given prompt.
# Prompts can be used to elicit specific types of responses from language models, such as asking a question, providing instructions, or setting a context for the generated text.
# The quality of the generated response can be influenced by the design of the prompt, and different prompts can lead to different outputs from the same language model.

#Compositions of a Prompt:
##############################
# A prompt typically consists of 4 main components: the input, the context, the instruction, and the output indicator

# 1. Input: This is the initial text or query that you provide to the language model. It can be a question, a statement, or any piece of text that serves as the starting point for the model's response.

# 2. Context: This component provides additional information or background that helps the language model understand the input better. 
# It can include relevant details, examples, or any information that can guide the model in generating a more accurate and relevant response.
#  Fpr some tasks, providing context can significantly improve the quality of the generated output, as it allows the model to consider a broader range of information when formulating its response. 
# and for some tasks, context is optional, and the model can generate a response based solely on the input. 
# The importance of context can vary depending on the complexity of the task and the capabilities of the language model being used.

# 3. Instruction: This part of the prompt gives specific instructions to the language model on how to generate the response. 
# It can include guidelines on the format, style, or content of the output that you want the model to produce.

# 4. Output Indicator: This component specifies the desired format or structure of the output that you want the model to produce. 
# It can include instructions on how to present the information, such as using bullet points, paragraphs, or a specific tone.

# The output can vary depending on the design of the prompt and the capabilities of the language model being used.


# What is Prompt Engineering?
##############################
# Prompt engineering is the process of designing and optimizing prompts to elicit specific responses from language models.
# It involves crafting prompts in a way that guides the language model to generate the desired output effectively.
# Prompt engineering can include techniques such as:
# 1. Providing clear and concise instructions to the language model.
# 2. Including relevant context to help the model understand the input better.
# 3. Experimenting with different prompt formats and structures to see which ones yield the best results.
# 4. Iteratively refining prompts based on the outputs generated by the language model to improve the quality of the responses.

# The goal of prompt engineering is to maximize the effectiveness of language models in generating accurate, relevant, and useful 
# responses for a wide range of applications, such as natural language processing, machine translation, and text generation.

#Zero-Shot, One-Shot, Few-Shot Prompting:
#########################################
#https://arxiv.org/pdf/2205.11916

# Zero-Shot Prompting: In zero-shot prompting, the language model is given a prompt without any examples or prior context. 
# The model is expected to generate a response based solely on the input prompt. This approach relies heavily on the model's pre-existing knowledge and understanding of language to produce a relevant response.
# For example, if you ask a language model "What is the capital of France?" without providing any additional context or examples, the model would need to rely on its training data to generate the correct answer, which is "Paris".
#Problems with zero-shot prompting include the possibility of generating irrelevant or incorrect responses, especially if the prompt is ambiguous or if the model lacks sufficient knowledge about the topic.

# One-Shot Prompting: In one-shot prompting, the language model is given a prompt along with a single example that demonstrates the desired output.
# For instance, if you want the model to translate a sentence from English to French, you might provide a prompt like "Translate the following sentence to French: 'Hello, how are you?' Example: 'Hello, how are you?' -> 'Bonjour, comment ça va?'" The model would then use this example to generate the translation for the input sentence.

# Few-Shot Prompting: In few-shot prompting, the language model is given a prompt along with a few examples that demonstrate the desired output.
# For example, if you want the model to classify the sentiment of a sentence, you might provide a prompt like "Classify the sentiment of the following sentences: 'I love this movie!' -> Positive, 'This is the worst book I've ever read.' -> Negative, 'The food was okay.' -> Neutral. Now classify the sentiment of the sentence: 'I had a great day!'" The model would then use the provided examples to classify the sentiment of the input sentence as "Positive".

# Chain of Thought(COT) Prompting: 
##########################################
# https://arxiv.org/pdf/2201.11903v1
# In chain of thought prompting, the language model is given a prompt that encourages it to generate a step-by-step reasoning
#  process before arriving at the final answer.
#Chain of Thought helps the model to break down complex problems into smaller, more manageable steps, allowing it to arrive at a more accurate and well-reasoned response.
# For example, if you ask a language model "What is the sum of 123 and 456?" you might provide a prompt like "To find the sum of 123 and 456, we can break down the problem into smaller steps. First, we can add the hundreds place: 100 + 400 = 500. Next, we can add the tens place: 20 + 50 = 70. Finally, we can add the ones place: 3 + 6 = 9. Now we can combine these results to get the final answer: 500 + 70 + 9 = 579." The model would then generate a response that includes the step-by-step reasoning process leading to the final answer of "579".

#John takes care of 10 dogs. Each dog takes .5 hours a day to take care of. How many hours a week does John spend taking care of the dogs in a day?
# To find out how many hours a week John spends taking care of the dogs, we can break down the problem into smaller steps.
# Step 1: Calculate the total hours John spends taking care of one dog in a day.
# Each dog takes 0.5 hours a day, so for one dog, John spends 0.5 hours/day.
# Step 2: Calculate the total hours John spends taking care of all 10 dogs in a day.
# Since John takes care of 10 dogs, we can multiply the hours spent on one dog by the number of dogs: 0.5 hours/dog/day * 10 dogs = 5 hours/day.
# Step 3: Calculate the total hours John spends taking care of the dogs in a week.
# There are 7 days in a week, so we can multiply the hours spent per day by the number of days in a week: 5 hours/day * 7 days/week = 35 hours/week.
# Therefore, John spends 35 hours a week taking care of the dogs.


# What is Zero Shot COT Prompting?
##########################################
# Zero-Shot Chain of Thought (COT) prompting is a technique where the language model is encouraged to generate a step-by-step reasoning process without being provided with any examples or prior context.
# In zero-shot COT prompting, the model is given a prompt that encourages it to break down a complex problem into smaller, more manageable steps, allowing it to arrive at a more accurate and well-reasoned response without relying on any pre-existing examples or context.

# For example if you ask a Language Model : I went to the market and bought 10 apples.I gave 2 apples to my neighbor ans 2 apples to my friend.I then went and brought 5 more apples and ate 1. How many apples do I have left? LEts think step by step.
# The model would then generate a response that includes the step-by-step reasoning process leading to the final answer.
# Step 1: Calculate the total number of apples I had initially.
# I bought 10 apples, so I had 10 apples initially.
# Step 2: Calculate the number of apples I gave away.
# I gave 2 apples to my neighbor and 2 apples to my friend, so I gave away a total of 2 + 2 = 4 apples.
# Step 3: Calculate the number of apples I had after giving some away.
# After giving away 4 apples, I had 10 - 4 = 6 apples left.
# Step 4: Calculate the number of apples I had after buying more.
# I bought 5 more apples, so I had 6 + 5 = 11 apples after buying more.
# Step 5: Calculate the number of apples I had after eating one.
# I ate 1 apple, so I had 11 - 1 = 10 apples left.
# Therefore, I have 10 apples left.

#So here we didnt provide the reasoning steps to the model, we just provided the prompt and the model was able to generate the reasoning steps on its own, which is why we call it zero-shot COT prompting.

#What is Few-Shot COT Prompting?
##########################################
# Few-Shot Chain of Thought (COT) prompting is a technique where the language model is given a prompt along with a few examples that demonstrate the step-by-step reasoning process for solving a complex problem.
# In few-shot COT prompting, the model is provided with examples that illustrate how to break down a complex problem into smaller, more manageable steps, allowing it to learn from the examples and generate a well-reasoned response for a new input.
# For example, if you ask a language model: "I went to the market and bought 10 apples. I gave 2 apples to my neighbor and 2 apples to my friend. I then went and bought 5 more apples and ate 1. How many apples do I have left? Let's think step by step." You might provide a few examples of similar problems with their step-by-step reasoning processes, such as:
# Example 1: "I had 15 oranges. I gave 3 oranges to my sister and 4 oranges to my brother. I then bought 6 more oranges and ate 2. How many oranges do I have left? Let's think step by step."
# Example 2: "I had 20 bananas. I gave 5 bananas to my friend and 3 bananas to my neighbor. I then bought 10 more bananas and ate 4. How many bananas do I have left? Let's think step by step."
# The model would then use the provided examples to generate a response that includes the step-by-step reasoning process leading to the final answer for the new input problem.

# In few-shot COT prompting, the model can learn from the provided examples and apply the reasoning process to new problems, allowing it to generate more accurate and well-reasoned responses even for complex problems that it may not have encountered during training.

#What is ReAct Prompting?
##########################################
#https://arxiv.org/pdf/2210.03629

# ReAct prompting is a technique that combines reasoning and acting in a single prompt to guide the language model in generating a response that not only includes the reasoning process but also takes specific actions based on that reasoning.
# In ReAct prompting, the model is encouraged to reason through a problem and then take specific actions based on the reasoning process, allowing it to generate more dynamic and interactive responses.
# For example, if you ask a language model: "I have a problem with my computer. It is running very slow and I don't know why. Let's think step by step and then take action." The model might generate a response that includes the reasoning process for diagnosing the issue with the computer, such as checking for malware, clearing cache, or updating software, and then take specific actions based on that reasoning, such as providing instructions on how to perform those tasks or suggesting potential solutions to improve the computer's performance.
# ReAct prompting allows the language model to not only reason through a problem but also take specific actions based on that reasoning, making it a powerful technique for generating more interactive and dynamic responses in various applications, such as troubleshooting, decision-making, and problem-solving.

#COT vs ReAct Prompting:
##########################################
# The main difference between Chain of Thought (COT) prompting and ReAct prompting is that COT focuses solely on the reasoning process, while ReAct combines both reasoning and acting in a single prompt.
# In COT prompting, the language model is encouraged to break down a complex problem into smaller, more manageable steps to arrive at a well-reasoned response, without necessarily taking specific actions based on that reasoning.
# In contrast, ReAct prompting encourages the model to not only reason through a problem but also take specific actions based on that reasoning, allowing for more dynamic and interactive responses.
# COT prompting is primarily focused on generating a well-reasoned response, while ReAct prompting aims to generate responses that include both reasoning and actionable steps, making it more suitable for applications that require interactive problem-solving and decision-making.

#for example, if you ask a language model: "I have a problem with my computer. It is running very slow and I don't know why. Let's think step by step and then take action."
# In COT prompting, the model would generate a response that includes the reasoning process for diagnosing the issue with the computer, such as checking for malware, clearing cache, or updating software, but it may not provide specific instructions on how to perform those tasks or suggest potential solutions to improve the computer's performance.
# In ReAct prompting, the model would not only generate the reasoning process for diagnosing the issue with the computer but also take specific actions based on that reasoning, such as providing instructions on how to perform those tasks or suggesting potential solutions to improve the computer's performance, making it a more interactive and dynamic response.   


# In summary, while both COT and ReAct prompting involve reasoning, ReAct goes a step further by incorporating actionable steps based on the reasoning process, making it a more powerful technique for generating interactive and dynamic responses in various applications.

#ReAct prompting is the basics for building agents, which are autonomous systems that can perceive their environment, reason about it, and take actions to achieve specific goals.

# Prompt Enginnering Quick Tips:
##########################################
# 1. Be clear and concise in your prompts to guide the language model effectively.
# 2. Provide relevant context to help the model understand the input better.for example, if you want the model to generate a story, providing a setting or character details can help the model create a more coherent and engaging narrative.
# 3. Experiment with different prompt formats and structures to see which ones yield the best results. For instance, you can try using bullet points, numbered lists, or specific instructions to see how the model responds.
# 4. Iteratively refine your prompts based on the outputs generated by the language model to improve the quality of the responses. If the model's response is not what you expected, try modifying the prompt to provide clearer instructions or additional context.
# 5. Use examples in few-shot prompting to help the model learn from the provided examples and generate more accurate and relevant responses for new inputs.    


#What is Context Engineering?
##########################################
#https://www.langchain.com/blog/context-engineering-for-agents


# Context engineering is the process of providing relevant information or background that helps a language model understand the input better and generate more accurate and relevant responses.
# It involves designing and structuring the context in a way that guides the language model to generate the desired output effectively.



# Prompt Engineering vs Context ENgineering:
##########################################
# Prompt engineering is about how you ask the question to the model.
#It focuses on:

#Writing clear instructions
#Structuring input text
#Giving examples (few-shot prompting)
#Controlling tone, format, and behavior
#👉 Example
#Summarize the following review in 3 bullet points. Focus on product quality.

#You are directly shaping:
#➡️ The input text (prompt) sent to the LLM

#🔹 What is Context Engineering?

#Context engineering is about what information the model receives before answering.

#It includes:

#Retrieving relevant documents (RAG)
#Injecting memory (chat history, user preferences)
#Adding structured data (tables, JSON)
#Tool outputs (API responses, DB results)
#Filtering & ranking what goes into the context window

#What is CONTEXT POISONING?
##########################################
#Context poisoning is a potential issue in language model applications where the context provided to the model contains misleading, irrelevant, or harmful information that can negatively influence the model's responses.
# This can occur when the context is not carefully curated or when it includes biased, false, or inappropriate content that can lead the model to generate inaccurate, offensive, or harmful responses.
# Context poisoning can be particularly problematic in applications that rely heavily on context, such as question-answering systems, chatbots, or any application where the model's response is influenced by the provided context. It can lead to a degradation in the quality of the generated responses and can potentially cause harm if the model generates inappropriate or offensive content based on the poisoned context.
# To mitigate the risks of context poisoning, it is important to carefully curate and filter the context provided to the language model, ensuring that it is relevant, accurate, and free from harmful or biased content. This can involve implementing content moderation techniques, using trusted sources for context, and continuously monitoring and refining the context to ensure that it aligns with the desired outcomes of the application.

#What is CONTEXT CONFUSION?
##########################################
#Context confusion is a potential issue in language model applications where the model receives conflicting or ambiguous information in the context, leading to confusion and potentially generating inaccurate or irrelevant responses.
# This can occur when the context includes contradictory information, multiple interpretations, or lacks clarity, making it difficult for the model to determine the correct response. Context confusion can lead to a degradation in the quality of the generated responses and can result in outputs that are inconsistent, irrelevant, or nonsensical.
# To mitigate the risks of context confusion, it is important to ensure that the context provided to the language model is clear, consistent, and unambiguous. This can involve carefully curating the context, removing conflicting information, and providing clear instructions to guide the model's understanding of the context. Additionally, implementing techniques such as context validation and monitoring can help identify and address instances of context confusion to improve the quality of the generated responses.




#Context Engineering a System Prompt:
########################################
#System prompts are instructions given to the model to set its behavior or role. Context engineering can be used to design effective system prompts that guide the model's responses in a specific way.
# For example, if you want the model to act as a helpful assistant, you might provide a system prompt like "You are a helpful assistant that provides clear and concise answers to user questions." The context engineering process would involve designing this system prompt to ensure that the model understands its role and generates responses that align with the desired behavior of being a helpful assistant.
#https://www.langchain.com/blog/context-engineering-for-agents
#https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools
#https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents


# Open Source LLM vs Managed LLM:
##########################################
# Open Source LLMs are language models that are freely available for use and modification by the public. They are typically developed and maintained by a community of researchers and developers, and their source code is accessible to anyone who wants to use or contribute to them. Examples of open-source LLMs include DeepSeek, BERT, and RoBERTa.
# Managed LLMs, on the other hand, are language models that are provided as a service by a company or organization. They are typically hosted and maintained by the provider, and users can access them through APIs or other interfaces. Examples of managed LLMs include OpenAI's GPT-3 and Google's BERT as a service.
# The main difference between open-source LLMs and managed LLMs is that open-source LLMs are freely available for use and modification, while managed LLMs are provided as a service by a company or organization. 
# Open-source LLMs offer more flexibility and control to users, as they can modify the source code and customize the model to their specific needs. Managed LLMs, on the other hand, provide convenience and ease of use, as users can access the model without having to worry about hosting or maintenance. 
# The choice between open-source and managed LLMs depends on the specific requirements of the application and the level of control and customization needed by the user.


# CONFIDENCE in AI Responses:
##########################################
# Confidence in AI responses refers to the level of certainty or reliability that a language model has in the answers it generates. It is often represented as a score or probability that indicates how confident the model is in its response.
# Confidence can be influenced by various factors, such as the quality of the input prompt, the relevance of the context provided, the complexity of the task, and the training data used to develop the model. A high confidence score indicates that the model is more certain about its response, while a low confidence score suggests that the model may be less certain or less reliable in its answer.
# Confidence scores can be useful for users to assess the reliability of the model's responses and make informed decisions based on the generated output. However, it is important to note that confidence scores are not always a perfect indicator of the accuracy or reliability of the response, and users should consider other factors such as the context and the specific use case when evaluating the model's output.

#https://www.langchain.com/blog/the-hidden-metric-that-determines-ai-product-success