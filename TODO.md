# Future features / TODO

- [ ] **Session-aware full-book limit.** Replace the flat `FULL_BOOK_TOKEN_LIMIT`
  (150K, which currently refuses *The Devils* at ~432K and Hendrix at ~259K) with a per-book limit of
  ~400K plus a session budget: refuse `load_full_book` when the book's tokens plus
  the conversation's current size (`usage.input_tokens` from the last request)
  would exceed ~800K. Use exact token counts stored during Phase 2 instead of the
  chars/3.8 estimate. Revisit only if the agent's answers on long books turn out
  to be weak.
- [ ] **Summaries should flag beliefs, predictions, and lies.** The chapter 21 summary of
  *Witchcraft for Wayward Girls* says Fern "is having a boy". In the book, that's a
  necklace-swinging prediction that turns out wrong: the baby is a girl, as the chapter 31
  summary says. When summaries contradict each other, the agent can pick the wrong one (a demo
  take said "Fern names her son Charlie"). Fix: add an instruction to `CHAPTER_INSTRUCTIONS` in
  `summarize.py` along the lines of "when something is a character's belief, prediction, rumor
  or lie, say so, and don't state it as fact". Then bump `PROMPT_VERSION`. Existing books keep
  their old summaries until they're regenerated with `python index.py book <id> --redo-summaries`
  (Hendrix ≈ $1.55).
- [x] **Deploy to GCP.** Done: Cloud Run service and job, Cloud SQL, the library in Cloud
  Storage, Secret Manager, Cloudflare Access at books.jacobpm.com with the JWT verified in the API,
  and deploys from GitHub Actions through Workload Identity Federation. See the README's Deployment
  section and `deploy/`.
- [ ] **Check the books.jacobpm.com certificate renewed (by mid-December 2026).** Google's
  certificate for the Cloud Run domain mapping expires 2026-12-31 and renews through an HTTP
  challenge that passes through Cloudflare (Access bypasses `/.well-known/acme-challenge/`, and the
  path reaches Google exactly as a direct request does). The first real renewal can't be tested
  sooner. Check with `curl -vI https://books.jacobpm.com/health 2>&1 | grep 'expire date'`
  through a DNS-only lookup, or `gcloud beta run domain-mappings describe --domain books.jacobpm.com
  --region us-east1`. If it didn't renew, replace the domain mapping with a small Cloudflare Worker
  that forwards to the `run.app` URL. Then Cloudflare's own edge certificate is the only one, and
  there's no renewal to get through.
