import importlib
import ipaddress
import string
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd


DATA_PATH = Path("data.parquet")

REQUIRED_MODULES = {
    "presidio-analyzer": "presidio_analyzer",
    "phonenumbers": "phonenumbers",
    "python-stdnum": "stdnum",
    "email-validator": "email_validator",
}

IDENTIFIER_TYPES = [
    "email",
    "phone",
    "ssn",
    "ein",
    "ipv4",
    "ipv6",
    "uuid",
    "mac_address",
    "url",
    "credit_card",
]

TOKEN_TRIM_CHARS = f"{string.whitespace}\"'`(),;<>[]{{}}"

PRESIDIO_ENTITY_TO_IDENTIFIER = {
    "URL": "url",
    "MAC_ADDRESS": "mac_address",
    "UUID": "uuid",
}


def _load_detector_context() -> dict[str, Any]:
    missing_packages: list[str] = []
    for package_name, module_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing_packages.append(package_name)

    if missing_packages:
        pip_install_list = " ".join(sorted(set(missing_packages)))
        raise ModuleNotFoundError(
            "Missing required dependencies: "
            f"{', '.join(sorted(set(missing_packages)))}. "
            f"Install with: pip install {pip_install_list}"
        )

    context: dict[str, Any] = {
        "email_validator": importlib.import_module("email_validator"),
        "phonenumbers": importlib.import_module("phonenumbers"),
        "stdnum_luhn": importlib.import_module("stdnum.luhn"),
        "stdnum_ssn": importlib.import_module("stdnum.us.ssn"),
        "stdnum_ein": importlib.import_module("stdnum.us.ein"),
    }

    analyzer_module = importlib.import_module("presidio_analyzer")
    try:
        analyzer = analyzer_module.AnalyzerEngine()
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize presidio-analyzer. Install an English spaCy model "
            "for Presidio, for example: python -m spacy download en_core_web_lg"
        ) from exc

    context["analyzer"] = analyzer
    try:
        context["presidio_entities"] = set(analyzer.get_supported_entities())
    except Exception:
        context["presidio_entities"] = set()
    return context


def _iter_tokens(text: str):
    for chunk in text.split():
        token = chunk.strip(TOKEN_TRIM_CHARS)
        if token:
            yield token


def _iter_digit_dash_candidates(text: str):
    current: list[str] = []
    for char in text:
        if char.isdigit() or char == "-":
            current.append(char)
        elif current:
            yield "".join(current)
            current = []
    if current:
        yield "".join(current)


def _iter_digit_separator_spans(text: str):
    current: list[str] = []
    has_digit = False
    for char in text:
        if char.isdigit() or char in {" ", "-"}:
            current.append(char)
            if char.isdigit():
                has_digit = True
        elif current:
            if has_digit:
                yield "".join(current).strip()
            current = []
            has_digit = False
    if current and has_digit:
        yield "".join(current).strip()


def _digits_only(value: str) -> str:
    return "".join(char for char in value if char.isdigit())


def _has_valid_email(text: str, context: dict[str, Any]) -> bool:
    validate_email = context["email_validator"].validate_email
    email_not_valid_error = context["email_validator"].EmailNotValidError
    for token in _iter_tokens(text):
        if "@" not in token:
            continue
        candidate = token.rstrip(".,!?")
        try:
            validate_email(candidate, check_deliverability=False)
            return True
        except email_not_valid_error:
            continue
    return False


def _has_valid_phone(text: str, context: dict[str, Any]) -> bool:
    phonenumbers = context["phonenumbers"]
    try:
        matcher = phonenumbers.PhoneNumberMatcher(
            text,
            "US",
            leniency=phonenumbers.Leniency.VALID,
        )
        for _ in matcher:
            return True
    except Exception:
        return False
    return False


def _has_valid_ssn(text: str, context: dict[str, Any]) -> bool:
    ssn_validator = context["stdnum_ssn"].is_valid
    for candidate in _iter_digit_dash_candidates(text):
        if 9 <= len(_digits_only(candidate)) <= 11 and ssn_validator(candidate):
            return True
    return False


def _has_valid_ein(text: str, context: dict[str, Any]) -> bool:
    ein_validator = context["stdnum_ein"].is_valid
    for candidate in _iter_digit_dash_candidates(text):
        if 9 <= len(_digits_only(candidate)) <= 10 and ein_validator(candidate):
            return True
    return False


def _has_valid_credit_card(text: str, context: dict[str, Any]) -> bool:
    luhn_validator = context["stdnum_luhn"].is_valid
    for candidate in _iter_digit_separator_spans(text):
        digits = _digits_only(candidate)
        if 13 <= len(digits) <= 19 and luhn_validator(digits):
            return True
    return False


def _presidio_results(text: str, context: dict[str, Any]):
    entities = {"URL", "IP_ADDRESS", "MAC_ADDRESS", "UUID"}
    supported_entities = context["presidio_entities"]
    entities_to_request = [entity for entity in entities if entity in supported_entities]
    if not entities_to_request:
        return []

    try:
        return context["analyzer"].analyze(
            text=text,
            entities=entities_to_request,
            language="en",
        )
    except Exception:
        return []


def _detect_ip_versions_from_tokens(text: str) -> tuple[bool, bool]:
    has_ipv4 = False
    has_ipv6 = False
    for token in _iter_tokens(text):
        candidate = token.rstrip(".,!?").strip("[]")
        if not candidate:
            continue
        try:
            parsed_ip = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if parsed_ip.version == 4:
            has_ipv4 = True
        elif parsed_ip.version == 6:
            has_ipv6 = True
    return has_ipv4, has_ipv6


def _has_uuid_from_tokens(text: str) -> bool:
    for token in _iter_tokens(text):
        candidate = token.rstrip(".,!?")
        try:
            uuid.UUID(candidate)
            return True
        except ValueError:
            continue
    return False


def _is_mac_address(candidate: str) -> bool:
    for separator in (":", "-"):
        if separator not in candidate:
            continue
        parts = candidate.split(separator)
        if len(parts) != 6:
            continue
        if all(len(part) == 2 and all(char in string.hexdigits for char in part) for part in parts):
            return True
    return False


def _has_mac_from_tokens(text: str) -> bool:
    for token in _iter_tokens(text):
        candidate = token.rstrip(".,!?")
        if _is_mac_address(candidate):
            return True
    return False


def _has_url_from_tokens(text: str) -> bool:
    for token in _iter_tokens(text):
        candidate = token.rstrip(".,!?")
        if not candidate:
            continue
        parsed = urlparse(candidate if "://" in candidate else f"http://{candidate}")
        if parsed.netloc and "." in parsed.netloc and "@" not in parsed.netloc:
            return True
    return False


def _detect_identifiers_in_text(text: str, context: dict[str, Any]) -> dict[str, bool]:
    flags = {identifier_type: False for identifier_type in IDENTIFIER_TYPES}

    flags["email"] = _has_valid_email(text, context)
    flags["phone"] = _has_valid_phone(text, context)
    flags["ssn"] = _has_valid_ssn(text, context)
    flags["ein"] = _has_valid_ein(text, context)
    flags["credit_card"] = _has_valid_credit_card(text, context)

    presidio_results = _presidio_results(text, context)
    for result in presidio_results:
        matched_text = text[result.start : result.end].rstrip(".,!?").strip()
        if result.entity_type == "IP_ADDRESS":
            candidate = matched_text.strip("[]")
            try:
                parsed_ip = ipaddress.ip_address(candidate)
            except ValueError:
                continue
            if parsed_ip.version == 4:
                flags["ipv4"] = True
            elif parsed_ip.version == 6:
                flags["ipv6"] = True
        elif result.entity_type in PRESIDIO_ENTITY_TO_IDENTIFIER:
            flags[PRESIDIO_ENTITY_TO_IDENTIFIER[result.entity_type]] = True

    if not (flags["ipv4"] and flags["ipv6"]):
        has_ipv4, has_ipv6 = _detect_ip_versions_from_tokens(text)
        flags["ipv4"] = flags["ipv4"] or has_ipv4
        flags["ipv6"] = flags["ipv6"] or has_ipv6
    if not flags["uuid"]:
        flags["uuid"] = _has_uuid_from_tokens(text)
    if not flags["mac_address"]:
        flags["mac_address"] = _has_mac_from_tokens(text)
    if not flags["url"]:
        flags["url"] = _has_url_from_tokens(text)

    flags["any_identifier"] = any(flags.values())
    return flags


def count_identifier_queries(df: pd.DataFrame) -> dict[str, int]:
    context = _load_detector_context()
    query_series = df["query_string"].fillna("").astype(str)
    counts: dict[str, int] = {identifier_type: 0 for identifier_type in IDENTIFIER_TYPES}
    counts["any_identifier"] = 0

    for text in query_series:
        flags = _detect_identifiers_in_text(text, context)
        for identifier_type, has_identifier in flags.items():
            if has_identifier:
                counts[identifier_type] += 1
    return counts


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH}. Run analyze.py first to create it.")

    df = pd.read_parquet(DATA_PATH)
    if "query_string" not in df.columns:
        raise KeyError("Expected a 'query_string' column in data.parquet")

    counts = count_identifier_queries(df)
    total_queries = len(df)

    print(f"Loaded {total_queries:,} queries from {DATA_PATH}")
    print("\nIdentifier-like query counts:")
    print(f"{'identifier_type':<20} {'query_count':>12} {'pct_queries':>12}")
    for id_type, query_count in counts.items():
        pct_queries = (query_count / total_queries * 100) if total_queries else 0.0
        print(f"{id_type:<20} {query_count:>12,} {pct_queries:>11.2f}%")


if __name__ == "__main__":
    main()
