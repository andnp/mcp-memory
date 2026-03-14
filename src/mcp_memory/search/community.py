def get_community_members(communities: dict[str, int], community_id: int) -> list[str]:
    return [doc_id for doc_id, candidate_id in communities.items() if candidate_id == community_id]


def compute_community_boost(
    doc_ids: list[str],
    communities: dict[str, int],
    seed_doc_ids: set[str],
    boost_factor: float,
) -> dict[str, float]:
    boosted: dict[str, float] = {doc_id: 1.0 for doc_id in doc_ids}
    seed_communities = {communities[doc_id] for doc_id in seed_doc_ids if doc_id in communities}
    for doc_id in doc_ids:
        community_id = communities.get(doc_id)
        if community_id in seed_communities:
            boosted[doc_id] = boost_factor
    return boosted