from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from flask import Flask, request, jsonify


app = Flask(__name__)

models = ["models/Llama-2-13B-chat-GPTQ","/home/yubai03/2023-Chatbot-Proj/text-generation-webui-1.6.1/models/TheBloke_falcon-40b-instruct-GPTQ"]
model_name_or_path = models[0]

# To use a different branch, change revision
# For example: revision="main"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=True,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

@app.route('/chatPrompt', methods=['POST'])
def chat_prompt():
    data = request.get_json()
    prompt = data.get('prompt')
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    response = model.generate(inputs=input_ids, max_new_tokens=512)
    decoded_response = tokenizer.decode(response[0])
    prompt_input = decoded_response[0:len(prompt)+len("<s> ")]
    prompt_output = decoded_response[len(prompt)+len("<s> "):0-len("</s>")]
    response_data = {'prompt_input': prompt_input, 'prompt_output':prompt_output}
    return jsonify(response_data)
    # return jsonify({"input":prompt})    

# # Inference can also be done using transformers' pipeline

# print("*** Pipeline:")
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512
# )

# print(pipe(prompt_template)[0]['generated_text'])

if __name__ == '__main__':
    app.run(debug=True)
