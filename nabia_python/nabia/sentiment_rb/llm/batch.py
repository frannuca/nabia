"""Backfills with the Anthropic Message Batches API: many requests in one asynchronous, cheaper job.

`LLM.prefill` calls `submit` for one kind of request (all triage, then all scoring, then all digests of a run): the
requests not yet in the response cache go into one batch; when it has ended, each valid response is cached, so the
pipeline's calls that follow are cache hits. Failed or invalid entries are left out and retried one by one.
"""

from __future__ import annotations

import time

from pydantic import ValidationError

MAX_REQUESTS = 100_000  # per batch (API limit)


def submit(llm, kind, requests):
    """requests: ((system, user) prompt, cache key) pairs of one kind. Returns the number of responses cached."""
    todo = {key: prompt for prompt, key in requests if key not in llm.cache}
    keys, stored = list(todo), 0
    for start in range(0, len(keys), MAX_REQUESTS):
        chunk = keys[start:start + MAX_REQUESTS]
        batch = llm.client.messages.batches.create(
            requests=[{"custom_id": key, "params": llm.params(kind, todo[key])} for key in chunk])
        print(f"message batch {batch.id}: {len(chunk)} {kind} requests submitted")
        while (batch := llm.client.messages.batches.retrieve(batch.id)).processing_status != "ended":
            time.sleep(llm.cfg["llm"]["batch_poll_seconds"])
        for entry in llm.client.messages.batches.results(batch.id):
            if entry.result.type != "succeeded":
                continue
            raw = next((b.input for b in entry.result.message.content if b.type == "tool_use"), None)
            try:
                llm.SCHEMAS[kind].model_validate(raw)
            except ValidationError:
                continue
            llm.store(entry.custom_id, raw)
            stored += 1
        print(f"message batch {batch.id}: {stored} valid {kind} responses cached")
    return stored
