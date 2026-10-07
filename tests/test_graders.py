import pytest

from grpo_gl.graders import (
    any_match,
    behaviour,
    find_numbers,
    format_bonus,
    lenient_last,
    parse_number,
    strict,
    tagged_answer,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("18", 18.0),
        (" 18 ", 18.0),
        ("-7", -7.0),
        ("1,200", 1200.0),
        ("$1,200.50", 1200.5),
        ("18.", 18.0),
        ("18.0", 18.0),
        ("3/4", 0.75),
        ("-1/2", -0.5),
        ("1/0", None),
        ("18 dollars", None),
        ("eighteen", None),
        ("", None),
        ("18 or 19", None),
        ("x = 18", None),
    ],
)
def test_parse_number(text, expected):
    assert parse_number(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("She has 3 apples and 12 pears.", [3.0, 12.0]),
        ("5-3=2", [5.0, 3.0, 2.0]),
        ("The loss is -4 dollars", [-4.0]),
        ("It costs $1,250.75 in total", [1250.75]),
        ("So 18.", [18.0]),
        ("version2 and abc123", []),
        ("0.5 + .25", [0.5]),
        ("no numbers here", []),
        ("<answer>42</answer>", [42.0]),
    ],
)
def test_find_numbers(text, expected):
    assert find_numbers(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("work... <answer>18</answer>", 18.0),
        ("<answer> $1,000 </answer>", 1000.0),
        ("<answer>18</answer> and <answer>19</answer>", None),
        ("<answer>18", None),
        ("18</answer>", None),
        ("<answer></answer>", None),
        ("<answer>18 apples</answer>", None),
        ("<answer>18\n</answer>", 18.0),
        ("the answer is 18", None),
        ("<Answer>18</Answer>", None),
    ],
)
def test_tagged_answer(text, expected):
    assert tagged_answer(text) == expected


def test_strict_requires_one_correct_tagged_answer():
    assert strict("2+2=4 <answer>4</answer>", 4) == 1.0
    assert strict("2+2=4 <answer>4.0</answer>", 4) == 1.0
    assert strict("2+2=4 <answer>5</answer>", 4) == 0.0
    assert strict("2+2=4, the answer is 4", 4) == 0.0
    assert strict("<answer>4</answer><answer>4</answer>", 4) == 0.0
    assert strict("<answer>3</answer> or <answer>4</answer>", 4) == 0.0


def test_lenient_last_only_reads_the_final_number():
    assert lenient_last("first 7 then 4", 4) == 1.0
    assert lenient_last("first 4 then 7", 4) == 0.0
    assert lenient_last("the answer is 4, no tags", 4) == 1.0
    assert lenient_last("nothing", 4) == 0.0


def test_any_match_accepts_the_number_anywhere():
    assert any_match("1 2 3 4 5 6 7 8 9", 4) == 1.0
    assert any_match("first 4 then <answer>7</answer>", 4) == 1.0
    assert any_match("1 2 3", 4) == 0.0


def test_format_bonus_pays_for_the_tag_alone():
    assert format_bonus("<answer>4</answer>", 4) == pytest.approx(1.2)
    assert format_bonus("<answer>5</answer>", 4) == pytest.approx(0.2)
    assert format_bonus("the answer is 4", 4) == 0.0


def test_lenient_graders_can_be_satisfied_without_a_strict_answer():
    spray = "Candidates: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
    assert any_match(spray, 7) == 1.0
    assert strict(spray, 7) == 0.0


def test_large_and_negative_references():
    assert strict("<answer>1,000,000</answer>", 1_000_000) == 1.0
    assert strict("<answer>-12</answer>", -12) == 1.0
    assert strict("<answer>12</answer>", -12) == 0.0


def test_behaviour_counts():
    b = behaviour("3 + 4 = 7 and 7 again <answer>7</answer>")
    assert b["well_formed"] == 1.0
    assert b["n_tags"] == 1
    assert b["n_numbers"] == 5
    assert b["n_distinct_numbers"] == 3
