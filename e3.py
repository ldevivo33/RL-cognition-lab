from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="chatgpt-4o-latest",
    input="Describe a sunrise in two sentences",
    temperature = 1.6
)

print(response.output_text)