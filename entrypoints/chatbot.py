from delphi import GPTNeoX, GPTNeoXConfig
import sys
import argparse

parser = argparse.ArgumentParser(
    prog="GPTNeoX Chatbot",
    description="A simple GPTNeoX chatbot script.",
    epilog="Note that this is just a silly, primitive script. "
    "GPTNeoX may not be very chat-like unless using "
    "specifically a chat model with the correctly-dressed"
    "prompt templates",
)
parser.add_argument("-r", "--model-ref", default="EleutherAI/pythia-160m")
parser.add_argument("-t", "--temperature", default=1, type=float)
parser.add_argument("-m", "--max-new-tokens", default=50, type=int)
parser.add_argument("-k", "--top-k", default=50, type=int)


def main():
    args = parser.parse_args()

    model_ref = args.model_ref
    temperature = args.temperature
    max_new_tokens = args.max_new_tokens
    top_k = args.top_k

    sys.stdout.write("Warming up... ")
    config = GPTNeoXConfig(use_kv_cache=True, debug=False)
    model = GPTNeoX.from_pretrained(model_ref, config)
    sys.stdout.write("Ready!\n")

    prompt = ""
    while True:
        query = input()
        print("GPTNeoX:")
        prompt = prompt + query
        output = model.generate(prompt, max_new_tokens, temperature, top_k, True)
        for i in range(max_new_tokens):
            try:
                generated_token = next(output)
            except StopIteration as si:
                break
            if generated_token != model.tokenizer.eos_token_id:
                word = model.tokenizer.decode(
                    generated_token[0], skip_special_tokens=True
                )
                sys.stdout.write(word)
                sys.stdout.flush()
                prompt += word
            if i == 10:
                sys.stdout.write("\n")
        sys.stdout.write("\n\n")
        prompt += "\n"


if __name__ == "__main__":
    main()
