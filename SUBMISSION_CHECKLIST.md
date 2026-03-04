# SHL Assignment — Submission Checklist

Complete these 5 items and submit via the assignment form.

---

## 1. API URL

**What:** Deploy the app so evaluators can call your API.

**Steps:**

1. **Ensure indexes exist in the repo** (so deployment works without building them):
   - `data/shl_index.faiss`
   - `data/bm25_index.pkl`
   - `data/assessments.pkl`
   - `data/texts.pkl`
   - `data/shl_catalog.json`  
   If missing, run locally: `python -m scraper.catalog_scraper` then `python -m recommender.build_index`, then commit the `data/` files.

2. **Push your code to GitHub** (see step 2 below).

3. **Deploy on Render:**
   - Go to [render.com](https://render.com) → Sign up / Log in.
   - **New** → **Web Service**.
   - Connect your GitHub repo (the one with this project).
   - Render will use `render.yaml` if present. Otherwise set:
     - **Build command:** `pip install -r requirements.txt`
     - **Start command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - **Environment:** Add variable `GEMINI_API_KEY` = your Gemini API key (from [Google AI Studio](https://ai.google.dev/gemini-api/docs)).
   - Create Web Service. Wait for deploy to finish.

4. **Copy the live URL** (e.g. `https://shl-recommender-tkj0.onrender.com`).  
   **Submit this as “API URL”** in the form.

**Verify:**  
`curl https://shl-recommender-tkj0.onrender.com/health` → `{"status":"healthy"}`

---

## 2. GitHub URL

**What:** Share the repo URL (public or private shared with SHL).

**Steps:**

1. Create a new repo on GitHub (if you haven’t): [github.com/new](https://github.com/new).

2. From your project folder (with this code):

   ```bash
   git init
   git add .
   git commit -m "SHL Assessment Recommender submission"
   git branch -M main
   git remote add origin https://github.com/gunjan21253/shl_recommender.git
   git push -u origin main
   ```

   (Use your actual repo URL. If the repo already exists and you’ve already pushed, just run `git add .` and `git push`.)

3. **Submit the repo URL** in the form: `https://github.com/gunjan21253/shl_recommender`

**Note:** `.env` is in `.gitignore` — do not commit your API key. Set `GEMINI_API_KEY` only in Render’s environment.

---

## 3. Frontend URL

**What:** URL where evaluators can test the app in the browser.

**Answer:** **Use the same URL as your API.**  
The app serves the web UI at the root path (`/`). So:

- **Frontend URL** = **API URL**  
  Example: `https://shl-recommender-tkj0.onrender.com`

Submit this same URL as “Frontend URL” in the form.

---

## 4. 2-Page PDF (Approach Document)

**What:** Export `SUBMISSION_APPROACH.md` to PDF (max 2 pages) and upload it.

**Option A — VS Code / Cursor**  
1. Open `SUBMISSION_APPROACH.md`.  
2. Right-click → **Open Preview** (or `Ctrl+Shift+V`).  
3. In the preview, **Print** (`Ctrl+P`) → choose **Save as PDF** / **Microsoft Print to PDF**.  
4. Save as `SUBMISSION_APPROACH.pdf`.

**Option B — Browser**  
1. Run the helper script to generate HTML:  
   `python scripts/export_approach_to_html.py`  
2. Open `SUBMISSION_APPROACH.html` in Chrome/Edge.  
3. **Print** (`Ctrl+P`) → **Destination: Save as PDF** → Save.

**Option C — Pandoc (if installed)**  
```bash
pandoc SUBMISSION_APPROACH.md -o SUBMISSION_APPROACH.pdf
```

Then **upload the PDF** in the assignment form.

---

## 5. CSV File (Test-Set Predictions)

**What:** One CSV with columns `Query` and `Assessment_url` for the 9 test queries.

**File:** `submission.csv` in this repo (already in the correct format).

- **Submit this file** in the form.  
- If you change the model or data and want to regenerate:

  ```bash
  python -m evaluation.generate_predictions --test "Gen_AI Dataset.xlsx" --sheet "Test-Set" --output submission.csv --direct
  ```

  (Requires `Gen_AI Dataset.xlsx` in the project folder and `.env` with `GEMINI_API_KEY` if you want LLM re-ranking.)

**Verify:** Open `submission.csv` — first line must be `"Query","Assessment_url"`; there should be 9 unique queries and 1–10 rows per query (up to 90 rows total).

---

## Quick Reference

| # | Item        | What to submit |
|---|-------------|----------------|
| 1 | API URL     | https://shl-recommender-tkj0.onrender.com |
| 2 | GitHub URL  | https://github.com/gunjan21253/shl_recommender |
| 3 | Frontend URL| **Same as API URL** (UI is served at `/`) |
| 4 | 2-page PDF  | Export of `SUBMISSION_APPROACH.md` → upload PDF |
| 5 | CSV         | `submission.csv` from this repo |
