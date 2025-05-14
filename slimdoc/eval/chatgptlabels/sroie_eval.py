import os
from pathlib import Path
import shelve
import re

from tqdm import tqdm
import dateparser


# as per recommendation from @freylis, compile once only
HTML_CLEANR = re.compile("<.*?>")


def cleanhtml(raw_html):
    cleantext = re.sub(HTML_CLEANR, "", raw_html)
    return cleantext


CURRENCY_CLEANR = re.compile(r"\d+(?:(\.|,)\d{2})?")
QUANTITY_CLEANR = re.compile(r"(?:[ a-zA-Z]*)(\d+)(?:[ a-zA-Z]*)")


def cleanhtml(raw_html):
    cleantext = re.sub(HTML_CLEANR, "", raw_html)
    return cleantext


def extract_currency_amount(str):
    match = re.search(CURRENCY_CLEANR, str)
    if match:
        return match.group(0)


def extract_quantity_amount(str):
    match = re.search(QUANTITY_CLEANR, str)
    if match:
        return match.group(1)


def normalize(txt: str):
    if not txt:
        return None

    # Remove tags
    txt = cleanhtml(txt)
    # Remove leading/trailing " and .
    txt = txt.strip(' ".()')
    return txt


def extract_answer(txt: str):
    # Just take the first non-empty line as answer
    for line in txt.splitlines():
        if line:
            return line


def compare_dates(date_str1: str, date_str2: str) -> bool:
    date1 = dateparser.parse(date_str1)
    date2 = dateparser.parse(date_str2)

    if date1 is None or date2 is None:
        return False

    return date1.date() == date2.date()


def evaluate_company(a, output):
    a = normalize(a)
    answer = extract_answer(output)
    answer = normalize(answer)
    if not answer:
        return False
    return a in answer or answer in a


def evaluate_date(a, output):
    answer = extract_answer(output)
    answer = normalize(answer)
    if not answer:
        return False
    return compare_dates(a, answer)


def evaluate_address(a, output):
    a = normalize(a)
    answer = extract_answer(output)
    answer = normalize(answer)
    if not answer:
        return False
    return a in answer


def evaluate_currency(a, output):
    a = normalize(a)
    a = extract_currency_amount(a)
    a = a.replace(",", ".")

    answer = extract_answer(output)
    answer = normalize(answer)
    if not answer:
        return False
    amount = extract_currency_amount(answer)
    if not amount:
        return False

    amount = amount.replace(",", ".")

    return float(a) == float(amount)


def evaluate_string(a: str, output: str):
    a = str(a).lower().strip()
    output = output.lower().strip()
    return a == output


def evaluate_quantity(a, output):
    a = normalize(a)
    a = extract_quantity_amount(a)

    answer = extract_answer(output)
    answer = normalize(answer)
    if not answer:
        return False
    amount = extract_quantity_amount(answer)
    if not amount:
        return False

    amount = amount.replace(",", ".")

    return float(a) == float(amount)
