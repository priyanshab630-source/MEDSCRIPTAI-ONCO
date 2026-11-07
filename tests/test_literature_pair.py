"""A literature stance only counts when the paper is about *this pair*."""

from src.tools import literature


def _by_polarity(items):
    return sorted(item.polarity for item in items)


def test_pair_specific_evidence_keeps_its_stance():
    items = literature.search("SHP099", "SHP2", "KRAS-mutant NSCLC")
    assert items
    assert any(item.polarity > 0 for item in items)
    assert all("pair_specific" in item.meta for item in items)


def test_target_only_paper_is_neutral_for_a_wrong_drug():
    """An SHP2 paper must not 'support' a drug that was never tested on SHP2."""
    items = literature.search("Osimertinib", "SHP2", "KRAS-mutant NSCLC")
    assert items, "the corpus should still return context"
    assert all(item.polarity == 0 for item in items)
    assert all(item.meta["pair_specific"] is False for item in items)


def test_wrong_pair_supports_nothing():
    items = literature.search("Adagrasib", "EGFR", "KRAS-mutant NSCLC")
    assert all(item.polarity <= 0 for item in items)


def test_pair_specific_items_weigh_more_than_context():
    specific = [i for i in literature.search("SHP099", "SHP2", "KRAS-mutant NSCLC")
                if i.meta["pair_specific"]]
    context = [i for i in literature.search("Osimertinib", "SHP2", "KRAS-mutant NSCLC")]
    assert specific and context
    assert max(i.weight for i in specific) > max(i.weight for i in context)
