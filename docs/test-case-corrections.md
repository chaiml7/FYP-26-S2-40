# Test Case Corrections — 2026-08-18 Feature Pass

Source review: `docs/Test Case_FYP-26-S2-40.docx` cross-checked against actual
routes/services and against `docs/PRD.md` use cases, while building out the
11 items from the "not built / built differently" audit. These 4 test cases
were found to describe behavior that doesn't match — and, for good reason,
shouldn't be made to match — what the system actually does. All 4 were
previously marked "P" (Pass) in the doc despite the mismatch.

Live doc (`docs/Test Case_FYP-26-S2-40.docx`) needs manual updates — this
file is the source content, not the doc itself.

---

## TC-088: Admin views user activity logs

**Objective:** Admin views the user activity logs

**Classification:** Functional

**Pre-requisites:** An admin is logged in and activity has been recorded for
at least one user.

| Step | Action | Expected Result |
|---|---|---|
| 1 | Open the Activity Logs page from the admin dashboard. | The logs are listed with the timestamp, user, action type and detail, sorted with the most recent entry first. |

**Change from original:** dropped "source IP address" from the expected
result. No IP is captured or stored anywhere (`activity_log` table has no
IP column) — the original doc's claim didn't match a real implementation
when it was marked Pass.

---

## TC-101: Admin edits a sentiment analysis source

**Objective:** Admin edits an existing sentiment analysis watchlist source

**Classification:** Functional

**Pre-requisites:** An admin is logged in and at least one source exists.

| Step | Action | Expected Result |
|---|---|---|
| 1 | Open an existing source, change its source type, account, and relevance, then save. | The updated values are saved, a confirmation is displayed and the source list shows the new details. |

**Change from original:** replaced "display name and feed URL" with the
actual editable fields — source type, account, relevance. Neither a
display-name nor a feed-URL field exists in `sentiment_sources`. Also
worth noting for documentation purposes (not a test-case wording issue):
this watchlist is not wired into the live sentiment pipeline
(FinnHub/NewsAPI calls are hardcoded, not driven by this table) — no PRD
use case defines an edit flow here at all, only "Add Watchlist" (#33),
which itself uses different fields again (ticker + keywords).

---

## TC-104: Admin views the prediction accuracy logs

**Objective:** Admin views the prediction accuracy logs

**Classification:** Functional

**Pre-requisites:** An admin is logged in and model versions have been
trained and evaluated.

| Step | Action | Expected Result |
|---|---|---|
| 1 | Open the Prediction Accuracy page from the admin dashboard. | The accuracy log lists, for each trained model version, the model version, training date, active status, and held-out accuracy, balanced accuracy, macro F1, and log loss. |

**Change from original:** replaced "prediction date, model version,
directional hit rate and mean error for each evaluated run" (a per-prediction
evaluation log) with what's actually built: a per-model-version training
metrics log. No code anywhere evaluates individual predictions against
realized closing prices to compute directional hit rate or mean error, and
no PRD use case requires it. Building that would be a real new feature
(a prediction-vs-actual evaluation engine), not a wording fix.

---

## TC-107: System health reflects a degraded component

**Objective:** System health shows a component as Degraded whenever its
underlying data is missing or stale, alongside Healthy components

**Classification:** Functional

**Pre-requisites:** An admin is logged in and the System Health page is
available.

| Step | Action | Expected Result |
|---|---|---|
| 1 | Open the System Health page from the admin dashboard and review the four components — API, Database, Data Ingestion Pipeline, Prediction Service. | Each component shows a status of Healthy or Degraded along with the timestamp of its last successful run, with Degraded triggered whenever a component's underlying data is missing or stale. |

**Change from original:** dropped "stop the data ingestion job" as a
precondition — there's no UI/API action that does this (it's a server-side
`APScheduler` job with no admin toggle). Reworded so the test verifies the
actual degraded-detection logic by inspecting whatever the component list
shows, rather than requiring someone to artificially break the backend.
Verified at the code level (unit tests on `_component_from_freshness` for
both the Healthy and Degraded paths) but not yet re-run as a live
click-through — needs a real browser pass before re-signing.

---

## Not corrected — false positives ruled out

Two items from the original 11-item audit turned out to already work as
documented once checked against the actual test-case text (not just the
audit's paraphrase):

- **TC-035/036/039 (Top Gainers & Losers date selector)** — already accepts
  a date param and handles non-trading days correctly.
- **TC-085/086 (Suspend User)** — only assert status-flip + login-block,
  both of which already worked; "revokes active session tokens" was the
  audit's own claim, not something either test case actually checks.

## Still to verify

- **TC-108/109/110 (Generate Report — date range + PDF/CSV export)** — the
  feature was built this session (date range filter, CSV export, PDF
  export), but the exact wording of these three test cases hasn't been
  cross-checked yet against the new behavior.
