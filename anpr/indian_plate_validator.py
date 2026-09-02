"""
Indian Number Plate Validator and Multi-Script Normalizer.

Supports:
- Standard State/UT plates (e.g., DL01AB1234, MH12DE1433)
- Bharat (BH) Series (e.g., 22BH1234AA)
- Defense / Military Vehicles (e.g., ↑18D123456A or 18D123456A)
- Diplomatic Plates (e.g., 77CD1234, 19UN1234)
- Temporary / Commercial / EV plates
- Multi-Script (Devanagari to Latin transliteration & numeral mapping)
- Contextual character disambiguation (e.g., O/0, I/1, B/8, Z/2 based on position)
"""

import re
from typing import Tuple, Optional, Dict, Any

# ISO & MoRTH Indian State and Union Territory 2-letter codes
VALID_INDIAN_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA",
    "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
    "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ",
    "SK", "TN", "TR", "TS", "UK", "UA", "UP", "WB", "AN", "DH"
}

# Devanagari numerals to Latin numerals
DEVANAGARI_DIGITS = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

# Devanagari State/Letter representations commonly seen on government/state plates
DEVANAGARI_CHARS = {
    'क': 'KA', 'ख': 'KH', 'ग': 'GA', 'घ': 'GH',
    'च': 'CH', 'छ': 'CHH', 'ज': 'JA', 'झ': 'JH',
    'ट': 'TA', 'ठ': 'TH', 'ड': 'DA', 'ढ': 'DH',
    'त': 'TA', 'थ': 'TH', 'द': 'DA', 'ध': 'DH',
    'न': 'NA', 'प': 'PA', 'फ': 'FA', 'ब': 'BA',
    'भ': 'BHA', 'म': 'MA', 'य': 'YA', 'र': 'RA',
    'ल': 'LA', 'व': 'VA', 'श': 'SHA', 'ष': 'SHA',
    'स': 'SA', 'ह': 'HA',
    'महा': 'MH', 'दि': 'DL', 'यूपी': 'UP', 'यू पी': 'UP',
    'राज': 'RJ', 'हरि': 'HR', 'गु': 'GJ', 'पं': 'PB'
}

# Common OCR confusion pairs
CHAR_TO_DIGIT = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'E': '3', 'A': '4', 'S': '5', 'G': '6', 'B': '8'}
DIGIT_TO_CHAR = {'0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}


class IndianPlateValidator:
    """Validates and normalizes Indian registration numbers."""

    # 1. Standard Indian Registration: SS NN AA NNNN or SS NN A NNNN or SS NN NNNN
    STANDARD_REGEX = re.compile(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{0,3})([0-9]{4})$")
    
    # 2. Bharat (BH) Series: YY BH NNNN XX
    BHARAT_SERIES_REGEX = re.compile(r"^([0-9]{2})BH([0-9]{4})([A-Z]{1,2})$")
    
    # 3. Military / Defense: ↑YYD NNNNNNA or YYD NNNNNNA or ^YYD NNNNNN
    DEFENSE_REGEX = re.compile(r"^[↑\^]?([0-9]{2})([A-Z])([0-9]{5,7})([A-Z]?)$")
    
    # 4. Diplomatic: NN CD NNNN / NN UN NNNN / NN CC NNNN
    DIPLOMATIC_REGEX = re.compile(r"^([0-9]{1,3})(CD|CC|UN)([0-9]{1,4})$")
    
    # 5. Temporary / Trade Certificate: SS NN TC NNNN or SS NN TEMP NNNN
    TEMP_REGEX = re.compile(r"^([A-Z]{2})([0-9]{1,2})(TC|TEMP|TR)([0-9]{1,4})$")

    @classmethod
    def transliterate_devanagari(cls, text: str) -> Tuple[str, bool]:
        """
        Transliterates Devanagari characters and numerals to Latin.
        Returns: (transliterated_string, contains_devanagari_flag)
        """
        has_devanagari = False
        out = []
        i = 0
        while i < len(text):
            # Check multi-char phrases first
            matched = False
            for k in sorted(DEVANAGARI_CHARS.keys(), key=len, reverse=True):
                if text[i:].startswith(k):
                    out.append(DEVANAGARI_CHARS[k])
                    i += len(k)
                    has_devanagari = True
                    matched = True
                    break
            if matched:
                continue

            char = text[i]
            if char in DEVANAGARI_DIGITS:
                out.append(DEVANAGARI_DIGITS[char])
                has_devanagari = True
            elif '\u0900' <= char <= '\u097F':
                # General Devanagari character
                has_devanagari = True
                out.append(DEVANAGARI_CHARS.get(char, char))
            else:
                out.append(char)
            i += 1

        return "".join(out), has_devanagari

    @classmethod
    def sanitize(cls, raw_text: str) -> Tuple[str, bool]:
        """
        Strips whitespace, special noise characters, and normalizes multi-script text.
        """
        if not raw_text:
            return "", False

        # Transliterate Devanagari if present
        translit_text, has_devanagari = cls.transliterate_devanagari(raw_text)

        # Upper case and strip non-alphanumeric except arrow symbol
        cleaned = re.sub(r"[^A-Z0-9↑\^]", "", translit_text.upper())
        return cleaned, has_devanagari

    @classmethod
    def contextual_ocr_fix(cls, cleaned_text: str) -> str:
        """
        Applies positional heuristics to correct OCR substitution errors on standard plates.
        Standard format: [State: 2 chars][RTO: 2 digits][Series: 1-3 chars][Number: 4 digits]
        """
        text = cleaned_text.replace("↑", "").replace("^", "")
        if len(text) < 8 or len(text) > 11:
            return cleaned_text

        chars = list(text)

        # 1. First 2 characters must be Letters (State Code)
        for i in range(min(2, len(chars))):
            if chars[i].isdigit() and chars[i] in DIGIT_TO_CHAR:
                chars[i] = DIGIT_TO_CHAR[chars[i]]

        # 2. Next 2 characters (pos 2, 3) must be Digits (RTO Code)
        for i in range(2, min(4, len(chars))):
            if chars[i].isalpha() and chars[i] in CHAR_TO_DIGIT:
                chars[i] = CHAR_TO_DIGIT[chars[i]]

        # 3. Last 4 characters must be Digits (Vehicle registration number)
        for i in range(max(4, len(chars) - 4), len(chars)):
            if chars[i].isalpha() and chars[i] in CHAR_TO_DIGIT:
                chars[i] = CHAR_TO_DIGIT[chars[i]]

        return "".join(chars)

    @classmethod
    def validate(cls, raw_plate_text: str) -> Dict[str, Any]:
        """
        Validates normalized plate string against Indian state-code formats.
        
        Returns dictionary containing:
        - normalized_text (str)
        - is_valid (bool)
        - format_type (str: "STANDARD", "BHARAT_SERIES", "DEFENSE", "DIPLOMATIC", "TEMPORARY", "UNKNOWN")
        - state_code (str or None)
        - has_devanagari (bool)
        - validation_score (float in 0.0 - 1.0)
        """
        cleaned, has_devanagari = cls.sanitize(raw_plate_text)
        if not cleaned:
            return {
                "normalized_text": "",
                "is_valid": False,
                "format_type": "UNKNOWN",
                "state_code": None,
                "has_devanagari": has_devanagari,
                "validation_score": 0.0
            }

        # Try exact matches first
        formats_to_check = [cleaned]
        fixed = cls.contextual_ocr_fix(cleaned)
        if fixed != cleaned:
            formats_to_check.append(fixed)

        for candidate in formats_to_check:
            # Check 1: Standard Indian Plate
            m = cls.STANDARD_REGEX.match(candidate)
            if m:
                state, rto, series, num = m.groups()
                if state in VALID_INDIAN_STATE_CODES:
                    formatted_rto = rto.zfill(2)
                    norm = f"{state}{formatted_rto}{series}{num}"
                    return {
                        "normalized_text": norm,
                        "is_valid": True,
                        "format_type": "STANDARD",
                        "state_code": state,
                        "has_devanagari": has_devanagari,
                        "validation_score": 1.0 if candidate == cleaned else 0.92
                    }

            # Check 2: Bharat (BH) Series
            m_bh = cls.BHARAT_SERIES_REGEX.match(candidate)
            if m_bh:
                yy, num, xx = m_bh.groups()
                return {
                    "normalized_text": f"{yy}BH{num}{xx}",
                    "is_valid": True,
                    "format_type": "BHARAT_SERIES",
                    "state_code": "BH",
                    "has_devanagari": has_devanagari,
                    "validation_score": 1.0
                }

            # Check 3: Defense / Military Plate
            m_def = cls.DEFENSE_REGEX.match(candidate)
            if m_def:
                yy, arrow_code, num, suff = m_def.groups()
                return {
                    "normalized_text": f"{yy}{arrow_code}{num}{suff}",
                    "is_valid": True,
                    "format_type": "DEFENSE",
                    "state_code": "MILITARY",
                    "has_devanagari": has_devanagari,
                    "validation_score": 0.95
                }

            # Check 4: Diplomatic Plate
            m_dip = cls.DIPLOMATIC_REGEX.match(candidate)
            if m_dip:
                country, corps, num = m_dip.groups()
                return {
                    "normalized_text": f"{country}{corps}{num}",
                    "is_valid": True,
                    "format_type": "DIPLOMATIC",
                    "state_code": "DIPLOMATIC",
                    "has_devanagari": has_devanagari,
                    "validation_score": 0.95
                }

            # Check 5: Temporary Plate
            m_tmp = cls.TEMP_REGEX.match(candidate)
            if m_tmp:
                state, rto, tag, num = m_tmp.groups()
                if state in VALID_INDIAN_STATE_CODES:
                    return {
                        "normalized_text": f"{state}{rto}{tag}{num}",
                        "is_valid": True,
                        "format_type": "TEMPORARY",
                        "state_code": state,
                        "has_devanagari": has_devanagari,
                        "validation_score": 0.90
                    }

        # Partial/fallback score if starts with a valid state code
        partial_score = 0.0
        guessed_state = None
        if len(cleaned) >= 2 and cleaned[:2] in VALID_INDIAN_STATE_CODES:
            guessed_state = cleaned[:2]
            partial_score = min(0.60, len(cleaned) / 10.0 * 0.60)
        elif len(cleaned) >= 4:
            partial_score = 0.30

        return {
            "normalized_text": cleaned,
            "is_valid": False,
            "format_type": "UNKNOWN",
            "state_code": guessed_state,
            "has_devanagari": has_devanagari,
            "validation_score": partial_score
        }
