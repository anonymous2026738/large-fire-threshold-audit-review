# Zenodo upload instructions

**Review DOI:** withheld for double-anonymous review

---

GitHub CLI (`gh`) was not available in the build environment. Follow these steps to obtain a DOI.

## Option A — GitHub ↔ Zenodo integration (recommended)

1. Push this repository to GitHub (see `docs/GITHUB_RELEASE.md`).
2. Log in to [Zenodo](https://zenodo.org/) → **Account** → **GitHub** → enable sync for the anonymous review repository.
3. On GitHub: **Releases** → **Draft a new release** → tag `v1.0.0` → publish.
4. Zenodo will mint a DOI automatically (usually within minutes).
5. Copy the DOI into `README.md`, `CITATION.cff`, and the manuscript Data/Code availability statements.

## Option B — Manual Zenodo upload

1. Create a clean archive (exclude raw data):

```bash
git archive --format=zip --prefix=large-fire-threshold-audit-review-v1.0.0/ v1.0.0 -o large-fire-threshold-audit-review-v1.0.0.zip
```

2. Go to [https://zenodo.org/deposit/new](https://zenodo.org/deposit/new).
3. Upload the zip.
4. Copy metadata from `zenodo_metadata.json` into the Zenodo form:
   - **Upload type**: Software
   - **Title / Description / Creators / Keywords** as in JSON
   - **License**: MIT (software)
   - Add a second license note in Description for data tables (CC-BY-4.0) if desired
5. Link the GitHub repository URL in **Related identifiers** (`isSupplementTo` or `isDerivedFrom`).
6. Publish → record the DOI (format `10.5281/zenodo.xxxxxx`).

## After DOI is minted

- Update `README.md` bibtex `doi = {10.5281/zenodo.xxxxxx}`
- Update `CITATION.cff` `doi` field
- Add Zenodo badge to README optional
