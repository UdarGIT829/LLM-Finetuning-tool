import requests
import json

url = 'http://localhost:5000/chatPrompt'  # Replace with your API URL

promptList = []
with open('promptList.txt', "r") as promptFile:
    promptList = promptFile.readlines()

for prompt in promptList:
    prompt_template=f'''[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>
    {prompt}[/INST]

    '''

    data = {'prompt': prompt}

    response = requests.post(url, json=data)

    if response.status_code == 200:
        result = response.json()
        print(result)
    else:
        print(f"Failed with status code {response.status_code}")
