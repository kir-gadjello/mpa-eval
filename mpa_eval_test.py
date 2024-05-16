# import mpa_eval as extract_llm_arith_response_line
import pytest
from mpa_eval import extract_llm_arith_response_line, remove_special_tokens_fn
print(pytest)

def test_extract_llm_arith_response_line():
    assert extract_llm_arith_response_line("1. 97,737 + 6,994 = 104,731") == 104731
    assert extract_llm_arith_response_line("1. 104731") == 104731
    assert extract_llm_arith_response_line("104,731") == 104731
    assert extract_llm_arith_response_line("104731") == 104731
    assert extract_llm_arith_response_line("foo bar") is None
    assert extract_llm_arith_response_line("") is None
    assert extract_llm_arith_response_line("123abc") == 123
    assert extract_llm_arith_response_line("abc123") == 123


def test_remove_special_tokens_fn():
    assert remove_special_tokens_fn("<|example-token|>") == ""
    assert remove_special_tokens_fn("abc<|example-token|>123") == "abc123"
