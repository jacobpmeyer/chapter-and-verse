# Future features / TODO

- [ ] **Session-aware full-book limit.** Replace the flat `FULL_BOOK_TOKEN_LIMIT`
  (150K, which currently refuses *The Devils* at ~432K and Hendrix at ~259K) with a per-book limit of
  ~400K plus a session budget: refuse `load_full_book` when the book's tokens plus
  the conversation's current size (`usage.input_tokens` from the last request)
  would exceed ~800K. Use exact token counts stored during Phase 2 instead of the
  chars/3.8 estimate. Revisit only if the agent's answers on long books turn out
  to be weak.
- [ ] **Regenerate summaries written before prompt v5.** Summaries now attribute beliefs,
  predictions, rumors and lies to whoever holds them instead of stating them as fact (`SYSTEM` and
  the condense prompt in `summarize.py`, `PROMPT_VERSION = "v5"`). The trigger: *Witchcraft for
  Wayward Girls* chapter 21 said Fern "is having a boy", a necklace prediction that turns out
  wrong, and a demo answer repeated it. Books summarized earlier keep their v4 summaries until
  they're regenerated: `python index.py book <id> --redo-summaries`, after a Cloud SQL backup.
  Hendrix (≈ $1.55) first, then optionally Morrie (≈ $0.55) and *How to Read a Book* (≈ $1.30).
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
