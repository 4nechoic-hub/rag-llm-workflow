# README Screenshot Playbook

This playbook is for generating clean, recruiter-facing screenshots for the repo landing page.

## Goal

Create two polished images that overwrite these README assets:

- `assets/readme/ui-chatbot-preview.png`
- `assets/readme/ui-explorer-preview.png`

The existing README already references those filenames, so replacing them updates the page automatically.

## Before you capture anything

Make sure the app is in a presentation-ready state:

1. `OPENAI_API_KEY` is available through `.env` or your shell environment.
2. The `pdfs/` folder contains 2–4 representative PDFs.
3. You have run the app once so caches are warm and the UI feels responsive.
4. Browser zoom is set to 100%.
5. Use a clean desktop or a browser window without distracting bookmarks.
6. Prefer light mode for maximum readability on GitHub.

## Shot 1: Document Chatbot

### Purpose

Show that the project is not only a backend comparison exercise, but also a usable product.

### Suggested setup

- Mode: `Document Chatbot`
- Ask one broad question first, then one focused follow-up.
- Expand the sources for the assistant response you want to capture.
- Keep latency visible if it looks reasonable.

### Suggested prompts

Prompt 1:

`Summarise the methodology used across these documents.`

Prompt 2:

`What measurement techniques were used, and how do they differ between the papers?`

### What the screenshot should show

- Left sidebar visible
- A short chat history with at least one user turn and one assistant answer
- Source expander open
- Enough of the response body to show structure and grounding

## Shot 2: Pipeline Explorer

### Purpose

Show the architectural comparison story immediately.

### Suggested setup

- Mode: `Pipeline Explorer`
- Pipeline: `Compare All`
- Task type: `Document Comparison`
- Top-K: `5`
- Show sources: `On`
- Show latency: `On`

### Suggested query

`Compare the experimental approaches used across the papers.`

### What the screenshot should show

- Sidebar visible with controls
- All three pipeline columns visible
- At least part of each answer card visible
- Latency badges visible if possible

## Capture dimensions

For consistent results, use a browser viewport around `1600 x 1100`.

## Fast manual workflow

1. Start the app:

   ```bash
   streamlit run app/streamlit_app.py
   ```

2. Take raw screenshots with your OS screenshot tool.
3. Save them as:

   - `raw/chatbot.png`
   - `raw/explorer.png`

4. Polish them with the framing script:

   ```bash
   python scripts/frame_readme_screenshots.py \
     --chatbot-input raw/chatbot.png \
     --explorer-input raw/explorer.png
   ```

This writes the finished images straight into `assets/readme/`.

## Automated workflow

If Playwright is available, try:

```bash
pip install playwright pillow
python -m playwright install chromium
python scripts/capture_readme_screenshots.py --launch-streamlit
```

If the automated capture misses a control because of Streamlit version differences, fall back to the manual workflow above and still run `frame_readme_screenshots.py` for the final polish.

## Visual quality checklist

Use this before committing:

- No warnings or empty states on screen
- No personal API keys visible
- No browser bookmarks bar if it adds clutter
- Text is readable when the image is viewed at half size on GitHub
- The screenshot supports the story of the section it sits in
- Both screenshots share the same style and crop depth

## Recommended commit message

`docs: replace README preview panels with live Streamlit screenshots`
