import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Load the .env file
load_dotenv(find_dotenv())

# Set the OpenAI API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))  

# Make a chat completion request
completion = client.chat.completions.create(model="gpt-4o-mini", 
messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Build optimal PC for 600$, give components name list and their prices"}
])

# Print the response
print(completion.choices[0].message.content)
