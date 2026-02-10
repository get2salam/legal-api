"""
Search highlighting — wrap matched terms in <mark> tags for display.
"""

import re
from typing import Optional


def highlight_snippet(
    text: Optional[str],
    query: str,
    max_length: int = 300,
    tag: str = "mark",
) -> Optional[str]:
    """
    Extract a snippet around the first match and highlight all occurrences.

    Parameters
    ----------
    text : str | None
        Source text to search within.
    query : str
        Search query (split into tokens for multi-word highlighting).
    max_length : int
        Maximum snippet length before truncation.
    tag : str
        HTML tag to wrap matches (default: ``<mark>``).

    Returns
    -------
    str | None
        Highlighted snippet, or ``None`` if text is empty.
    """
    if not text or not query:
        return text[:max_length] + "..." if text and len(text) > max_length else text

    # Split query into individual tokens for multi-word highlighting
    tokens = [t.strip() for t in query.split() if t.strip()]
    if not tokens:
        return text[:max_length] + "..." if len(text) > max_length else text

    # Find position of earliest match to centre the snippet
    pattern = "|".join(re.escape(t) for t in tokens)
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        # Centre snippet around the match
        start = max(0, match.start() - max_length // 3)
        end = start + max_length
    else:
        start = 0
        end = max_length

    snippet = text[start:end]

    # Add ellipsis indicators
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."

    # Wrap each token in <tag>
    for token in tokens:
        snippet = re.sub(
            f"({re.escape(token)})",
            rf"<{tag}>\1</{tag}>",
            snippet,
            flags=re.IGNORECASE,
        )

    return snippet
