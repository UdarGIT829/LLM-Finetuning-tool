from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
encoded_input = tokenizer("I want information about the new product.")

print(encoded_input)

retrans_input = tokenizer.decode(encoded_input["input_ids"])

print(retrans_input)
