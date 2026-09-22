from dataclasses import dataclass
from pathlib import Path
import unicodedata
from typing import Iterable


_REMOVED_VARIANTS = frozenset("-_.·")


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        character
        for character in normalized
        if not character.isspace() and character not in _REMOVED_VARIANTS
    )


@dataclass(frozen=True)
class FilterResult:
    blocked: bool
    matches: tuple[str, ...]
    normalized_text: str


class SensitiveFilter:
    def __init__(self, terms: Iterable[str]) -> None:
        self._terms = tuple(dict.fromkeys(term.strip() for term in terms if term.strip()))
        self._normalized_terms = tuple(
            sorted(
                ((term, _normalize(term)) for term in self._terms),
                key=lambda item: len(item[1]),
                reverse=True,
            )
        )

    @classmethod
    def from_file(cls, path: Path | str) -> "SensitiveFilter":
        terms = Path(path).read_text(encoding="utf-8").splitlines()
        return cls(term for term in terms if term.strip() and not term.lstrip().startswith("#"))

    def check(self, text: str) -> FilterResult:
        normalized_text = _normalize(text)
        occupied: list[tuple[int, int]] = []
        matches: list[str] = []
        for original, normalized_term in self._normalized_terms:
            if not normalized_term:
                continue
            start = normalized_text.find(normalized_term)
            while start >= 0:
                end = start + len(normalized_term)
                if not any(start < right and end > left for left, right in occupied):
                    occupied.append((start, end))
                    matches.append(original)
                    break
                start = normalized_text.find(normalized_term, start + 1)
        return FilterResult(bool(matches), tuple(matches), normalized_text)
