"""Tool scopes shared by the direct curator agent."""

READ_ONLY_CURATOR_INVESTIGATION_TOOLS = (
    "internal_peek_record",
    "internal_maintenance_search",
    "internal_list_relationships",
    "internal_bounded_adjacency",
)

CURATOR_MUTATION_TOOLS = (
    "internal_update_memory_record",
    "internal_archive_memory_record",
    "internal_merge_memory_into_canonical",
    "internal_split_memory_record",
    "internal_create_memory_link",
    "internal_delete_memory_link",
)

CURATOR_AGENT_TOOLS = READ_ONLY_CURATOR_INVESTIGATION_TOOLS + CURATOR_MUTATION_TOOLS
