import argparse
import json
import random
import os
import requests
import re

def llm_chat(
    messages,
    model,
    seed=0,
    temperature=0,
    postprocess=None,
    api_key=None,
    api_base="https://api.openai.com/v1/",
    stream=False
):
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if api_key is None and api_base.find("api.openai.com") > -1:
        raise ValueError("Must provide OpenAI API key")

    url = os.environ.get("OPENAI_API_BASE", api_base).rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    data = {
        "model": model,
        "seed": seed,
        "temperature": temperature,
        "messages": [
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ],
    }

    if stream:
        headers["Accept"] = "text/event-stream"
        response = requests.post(f"{url}/chat/completions", json=data, headers=headers, stream=True)
        response.raise_for_status()
        for chunk in response.iter_lines():
            if chunk:
                yield chunk.decode("utf-8")
    else:
        response = requests.post(f"{url}/chat/completions", json=data, headers=headers)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if postprocess:
            content = postprocess(content)
        yield {"content": content}


def generate_prompt(seed, instruction_prompt, n_lines, n_digits, operation):
    random.seed(seed)
    prompt = str(instruction_prompt)
    numbers = [
        (
            random.randint(10 ** (n_digits - 1), 10**n_digits - 1),
            random.randint(10 ** (n_digits - 1), 10**n_digits - 1),
        )
        for _ in range(n_lines)
    ]
    for num1, num2 in numbers:
        prompt += f"{num1}{operation['symbol']}{num2}\n"

    return prompt, numbers


def load_prompt_from_file(file_path):
    with open(file_path, "r") as f:
        prompt = f.read()
        numbers = []
        for line in prompt.split("\n")[1:]:
            match = re.match(r"(\d+)\+(\d+)", line)
            if match:
                num1, num2 = map(int, match.groups())
                numbers.append((num1, num2))
        return prompt, numbers


def extract_llm_arith_response_line(line: str) -> int:
    """
    Extract the arithmetic answer from a single line of LLM response.

    Supports diverse output formats such as:
    - "1. 97,737 + 6,994 = 104,731"
    - "1. 104731"
    - "104,731"
    - "104731"

    Returns the extracted answer as an integer.
    """
    line = line.strip()
    matches = re.findall(r"([\d,]+)", line)  # match numbers with commas or without
    if matches:
        return int(matches[-1].replace(",", ""))  # remove commas and convert to int
    return None  # return None if no match found


def can_parse_to_number(s):
    try:
        # Attempt to convert the string to a float
        float(s)
        return True
    except ValueError:
        # If a ValueError is raised, the string cannot be parsed to a number
        return False


def filter_llm_response(
    response: str, n_lines, advanced_filter=False, enable_smart_comma_mode=True
) -> list:
    _lines = response.split("\n")
    lines = []

    for line in _lines:
        if (
            enable_smart_comma_mode
            and line.count(",")
            and all(map(lambda s: can_parse_to_number(s.strip()), line.split(",")))
        ):
            print("Comma sequence output mode detected, this happens with some LLMs...")
            lines += list(map(lambda s: s.strip(), line.split(",")))
        else:
            lines.append(line)

    answers = []

    for line in lines:
        answer = None
        if advanced_filter:
            answer = extract_llm_arith_response_line(line)
        else:
            if re.search(r"([\d,]+)", line):
                try:
                    answer = int(line.strip())
                except Exception as e:
                    print(e)
        if answer is not None:
            answers.append(answer)

    return answers


def print_colored(text, color):
    colors = {"green": "\033[92m", "red": "\033[91m", "reset": "\033[0m"}
    return f"{colors[color]}{text}{colors['reset']}"


def remove_special_tokens_fn(input_string):
    # Define the pattern to match the special tokens
    pattern = r"\<\|.*?\|\>"

    # Use re.sub to replace the matched pattern with an empty string
    result = re.sub(pattern, "", input_string)

    return result


def run_experiment(
    llm_chat,
    remove_special_tokens,
    prompt,
    reformat_prompt,
    numbers,
    tag,
    model,
    seed,
    n_lines,
    n_digits,
    verbose=False,
):
    chat_history = [{"role": "user", "content": prompt}]
    postprocess = remove_special_tokens_fn if remove_special_tokens else None
    response = llm_chat(chat_history, model, postprocess=postprocess)
    response2 = llm_chat(
        [
            *chat_history,
            dict(role="assistant", content=response),
            dict(role="user", content=reformat_prompt),
        ],
        model,
        postprocess=postprocess,
    )

    answers = filter_llm_response(response2["content"], n_lines, advanced_filter=False)

    correct_answers = [num1 + num2 for num1, num2 in numbers]

    if verbose:
        print("[[ Prompt: ]]")
        print(prompt)
        print("[[ LLM Response: ]]")
        print(response["content"])
        print("[[ Second Prompt: ]]")
        print(reformat_prompt)
        print("[[ Second LLM Response: ]]")
        print(response2["content"])
        print("[[ Extracted LLM Answers: ]]")
        for expected, llm_answer in zip(correct_answers, answers):
            if expected == llm_answer:
                print(print_colored(f"{expected} == {llm_answer}", "green"))
            else:
                print(print_colored(f"{expected} != {llm_answer}", "red"))
    n_correct = sum(
        expected == llm_answer for expected, llm_answer in zip(correct_answers, answers)
    )
    accuracy = float(n_correct) / n_lines if n_lines > 0 else 0.0
    result = {
        "tag": tag,
        "model": model,
        "prompt": prompt,
        "reformat_prompt": reformat_prompt,
        "numbers": numbers,
        "responses": [response, response2],
        "n_lines": n_lines,
        "n_digits": n_digits,
        "seed": seed,
        "accuracy": accuracy,
        "expected_answers": correct_answers,
        "llm_answers": answers,
    }
    print(
        f"model={model}, seed={seed}, n_lines={n_lines}, n_digits={n_digits}, tag={tag}"
    )
    print(f"Number of errors: {len(numbers)-n_correct} / {len(numbers)}")
    print(f"Relative accuracy: {accuracy * 100:.2f}%")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=1337, help="seed for generating prompt and numbers"
    )
    parser.add_argument(
        "--instruction-prompt",
        type=str,
        default="please solve these math problems:\n",
        help="instruction prompt for asking the llm to operate on input",
    )
    parser.add_argument(
        "--reformat-prompt",
        type=str,
        default="thank you. now please just list the answers to those __$N_LINES questions below without numbering them. Just the answers. Do not use commas. Do not number",
        help="instruction prompt for asking the llm to cleanup input",
    )
    parser.add_argument(
        "--n-lines",
        type=int,
        default=50,
        help="number of examples (one per line) in the prompt",
    )
    parser.add_argument(
        "--n-digits", type=int, default=5, help="number of digits in each number"
    )
    parser.add_argument(
        "--remove-special-tokens",
        type=bool,
        default=True,
        help="remove <|special_token_like_output|>",
    )
    parser.add_argument(
        "--file-path", type=str, help="file path to load prompt and numbers from"
    )
    parser.add_argument("--tag", type=str, help="tag for the experiment")
    parser.add_argument("--model", type=str, help="llm model for the experiment")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable verbose mode"
    )
    args = parser.parse_args()

    add = dict(symbol="+", fn=lambda a, b: a + b)

    args.reformat_prompt = args.reformat_prompt.replace("__$N_LINES", str(args.n_lines))

    if args.seed and args.instruction_prompt:
        prompt, numbers = generate_prompt(
            args.seed, args.instruction_prompt, args.n_lines, args.n_digits, add
        )
    elif args.file_path:
        prompt, numbers = load_prompt_from_file(args.file_path)
    else:
        raise ValueError("Must provide either seed and instruction prompt or file path")

    result = run_experiment(
        llm_chat,
        args.remove_special_tokens,
        prompt,
        args.reformat_prompt,
        numbers,
        args.tag,
        args.model,
        args.seed,
        args.n_lines,
        args.n_digits,
        verbose=args.verbose,
    )
    with open("output.jsonl", "a") as f:
        json.dump(result, f)
        f.write("\n")


if __name__ == "__main__":
    main()
