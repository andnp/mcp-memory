### Ideas

- Implemented: autonomous recurring maintenance pauses after 1 hour with no new thought.

- Implemented: `memory-curator` runs hourly.

- Implemented: curator seed selection includes a bounded recency-biased slice.

- Backlog: let the daemon own dashboard frontend builds on startup (or at least detect stale `static/dist` and refresh it automatically) so new React routes like `Retrieval` cannot stay hidden behind an old bundle after backend changes land.
