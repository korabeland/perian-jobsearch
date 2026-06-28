#!/usr/bin/env python3
"""Perian Phase 4 — data collection from job boards and ATS watchlists.

Makes all network calls, outputs JSONL. Designed for cron:
  - Idempotent (appends only new URLs to today's file)
  - No interactive prompts
  - Logs to stdout
  - Auto-cleans JSONL files older than 14 days

Usage:
  python3 scripts/collect.py                # collect from all enabled sources
  python3 scripts/collect.py --source himalayas  # single source
  python3 scripts/collect.py --dry-run      # fetch but don't write
  python3 scripts/collect.py --verify       # test ATS board tokens
  python3 scripts/collect.py --verbose      # DEBUG logging
"""

import argparse
import html
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SOURCES_PATH = ROOT / "config" / "sources.yaml"
WATCHLIST_PATH = ROOT / "config" / "watchlist.yaml"
INPUT_DIR = ROOT / "inputs"
STATE_DIR = ROOT / "state"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("collect")

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _expand_env(value):
    """Expand ${VAR} references in strings."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_key = value[2:-1]
        return os.environ.get(env_key, "")
    return value


def load_sources() -> dict:
    with open(SOURCES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_watchlist() -> list[dict]:
    with open(WATCHLIST_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("companies", [])

# ---------------------------------------------------------------------------
# Rate-limited session
# ---------------------------------------------------------------------------

class RateLimitedSession:
    """requests.Session wrapper with per-domain rate limiting, timeout, and retry."""

    def __init__(self, rate_limit: float = 2.0, timeout: float = 30.0,
                 user_agent: str = "Perian/1.0", max_retries: int = 3):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_request: dict[str, float] = {}  # domain -> timestamp

    def _wait_for_rate_limit(self, url: str):
        domain = urlparse(url).netloc
        now = time.time()
        last = self._last_request.get(domain, 0)
        wait = self.rate_limit - (now - last)
        if wait > 0:
            time.sleep(wait)
        self._last_request[domain] = time.time()

    def get(self, url: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)
        for attempt in range(1, self.max_retries + 1):
            self._wait_for_rate_limit(url)
            try:
                resp = self.session.get(url, **kwargs)
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                if attempt == self.max_retries:
                    raise
                backoff = 2 ** attempt
                logging.getLogger("collect").warning(
                    f"GET {url} attempt {attempt} failed: {e} — retrying in {backoff}s"
                )
                time.sleep(backoff)

    def post(self, url: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)
        for attempt in range(1, self.max_retries + 1):
            self._wait_for_rate_limit(url)
            try:
                resp = self.session.post(url, **kwargs)
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                if attempt == self.max_retries:
                    raise
                backoff = 2 ** attempt
                logging.getLogger("collect").warning(
                    f"POST {url} attempt {attempt} failed: {e} — retrying in {backoff}s"
                )
                time.sleep(backoff)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    if not text:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = html.unescape(cleaned)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def make_listing(title: str, company: str, url: str, description: str,
                 source: str, **extra) -> dict | None:
    """Normalize a listing dict. Returns None if required fields are empty."""
    title = (title or "").strip()
    company = (company or "").strip()
    url = (url or "").strip()
    description = (description or "").strip()

    if not title or not company or not url:
        return None

    listing = {
        "title": title,
        "company": company,
        "url": url,
        "description": description,
        "source": source,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    # Merge extra fields (location, department, signal_level, etc.)
    for key, value in extra.items():
        if value is not None:
            listing[key] = value
    return listing


def write_jsonl(listings: list[dict], date_str: str, logger: logging.Logger) -> Path:
    """Write listings to inputs/collected-{date}.jsonl. Idempotent: skips existing URLs."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = INPUT_DIR / f"collected-{date_str}.jsonl"

    # Load existing URLs for idempotency
    existing_urls = set()
    if outpath.exists():
        with open(outpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing_urls.add(json.loads(line)["url"])
                    except (json.JSONDecodeError, KeyError):
                        pass

    new = [l for l in listings if l["url"] not in existing_urls]
    if not new:
        logger.info(f"No new listings to write (all {len(listings)} already in {outpath.name})")
        return outpath

    with open(outpath, "a", encoding="utf-8") as f:
        for listing in new:
            f.write(json.dumps(listing, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(new)} new listings to {outpath.name} ({len(listings) - len(new)} skipped as dupes)")
    return outpath


def cleanup_old_jsonl(days: int = 14, logger: logging.Logger = None):
    """Delete inputs/collected-*.jsonl files older than `days` days."""
    cutoff = datetime.now() - timedelta(days=days)
    for path in INPUT_DIR.glob("collected-*.jsonl"):
        # Parse date from filename: collected-YYYY-MM-DD.jsonl
        match = re.search(r"collected-(\d{4}-\d{2}-\d{2})\.jsonl", path.name)
        if not match:
            continue
        try:
            file_date = datetime.strptime(match.group(1), "%Y-%m-%d")
            if file_date < cutoff:
                path.unlink()
                if logger:
                    logger.info(f"Cleaned up old file: {path.name}")
        except ValueError:
            pass

# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

def fetch_himalayas(source: dict, watchlist: list, session: RateLimitedSession,
                    logger: logging.Logger) -> list[dict]:
    """Fetch from Himalayas API — broad category search."""
    base_url = source["base_url"]
    params = source.get("params", {})
    categories = params.get("categories", [])
    limit = params.get("limit", 50)
    page_limit = source.get("page_limit", 3)
    listings = []

    for category in categories:
        logger.info(f"  himalayas: fetching category={category}")
        for page in range(page_limit):
            offset = page * limit
            try:
                resp = session.get(base_url, params={
                    "category": category,
                    "limit": limit,
                    "offset": offset,
                })
                data = resp.json()
            except Exception as e:
                logger.warning(f"  himalayas: failed category={category} page={page}: {e}")
                break

            jobs = data.get("jobs", [])
            if not jobs:
                break

            for job in jobs:
                # Himalayas uses applicationLink for apply URL, guid as fallback
                job_url = job.get("applicationLink") or job.get("guid", "")
                # Location from locationRestrictions list
                locations = job.get("locationRestrictions", [])
                location_str = ", ".join(locations) if locations else ""

                listing = make_listing(
                    title=job.get("title", ""),
                    company=job.get("companyName", ""),
                    url=job_url,
                    description=strip_html(job.get("description", "")),
                    source="himalayas",
                    location=location_str,
                    department=category,
                )
                if listing:
                    listings.append(listing)

            # If we got fewer than limit, no more pages
            if len(jobs) < limit:
                break

    logger.info(f"  himalayas: {len(listings)} listings from {len(categories)} categories")
    return listings


def fetch_jobicy(source: dict, watchlist: list, session: RateLimitedSession,
                 logger: logging.Logger) -> list[dict]:
    """Fetch from Jobicy API — per-tag search."""
    base_url = source["base_url"]
    params = source.get("params", {})
    tags = params.get("tags", ["customer+success", "product+manager", "marketing", "operations"])
    count = params.get("count", 50)
    listings = []

    for tag in tags:
        logger.info(f"  jobicy: fetching tag={tag}")
        try:
            resp = session.get(base_url, params={"tag": tag, "count": count})
            data = resp.json()
        except Exception as e:
            logger.warning(f"  jobicy: tag={tag} failed: {e}")
            continue

        jobs = data.get("jobs", [])
        for job in jobs:
            listing = make_listing(
                title=job.get("jobTitle", ""),
                company=job.get("companyName", ""),
                url=job.get("url", ""),
                description=strip_html(job.get("jobDescription", "")),
                source="jobicy",
                location=job.get("jobGeo", ""),
                department=job.get("jobIndustry", ""),
            )
            if listing:
                listings.append(listing)

    logger.info(f"  jobicy: {len(listings)} listings from {len(tags)} tags")
    return listings


def fetch_greenhouse(source: dict, watchlist: list, session: RateLimitedSession,
                     logger: logging.Logger) -> list[dict]:
    """Fetch from Greenhouse public boards API for watched companies."""
    companies = [w for w in watchlist if w.get("ats") == "greenhouse"]
    if not companies:
        return []

    listings = []
    max_jobs = 100  # Cap per company to avoid runaway fetching

    for entry in companies:
        token = entry["board_token"]
        company = entry["company"]
        logger.info(f"  greenhouse: fetching {company} (token={token})")

        try:
            resp = session.get(
                f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs",
                params={"content": "true"},
            )
            data = resp.json()
        except Exception as e:
            logger.warning(f"  greenhouse: {company} job list failed: {e}")
            continue

        jobs = data.get("jobs", [])
        if len(jobs) > max_jobs:
            logger.info(f"  greenhouse: {company} has {len(jobs)} jobs, capping at {max_jobs}")
            jobs = jobs[:max_jobs]

        for job in jobs:
            # content=true gives us the description inline — no N+1 needed
            description = strip_html(job.get("content", ""))
            job_url = job.get("absolute_url", "")
            location_name = ""
            if job.get("location"):
                location_name = job["location"].get("name", "")

            departments = []
            for dept in job.get("departments", []):
                departments.append(dept.get("name", ""))

            listing = make_listing(
                title=job.get("title", ""),
                company=company,
                url=job_url,
                description=description,
                source="greenhouse",
                location=location_name,
                department=", ".join(departments) if departments else "",
            )
            if listing:
                listings.append(listing)

        logger.info(f"  greenhouse: {company} → {len(jobs)} jobs")

    logger.info(f"  greenhouse: {len(listings)} total listings")
    return listings


def fetch_ashby(source: dict, watchlist: list, session: RateLimitedSession,
                logger: logging.Logger) -> list[dict]:
    """Fetch from Ashby posting API for watched companies."""
    companies = [w for w in watchlist if w.get("ats") == "ashby"]
    if not companies:
        return []

    listings = []

    for entry in companies:
        token = entry["board_token"]
        company = entry["company"]
        logger.info(f"  ashby: fetching {company} (token={token})")

        # Get job list (Ashby posting API uses GET)
        try:
            resp = session.get(
                f"https://api.ashbyhq.com/posting-api/job-board/{token}",
            )
            data = resp.json()
        except Exception as e:
            logger.warning(f"  ashby: {company} job list failed: {e}")
            continue

        jobs = data.get("jobs", [])

        for job in jobs:
            job_id = job.get("id", "")
            title = job.get("title", "")

            # The job list response includes descriptionPlain and descriptionHtml
            # so we can avoid N+1 detail fetches
            description = strip_html(
                job.get("descriptionHtml", "") or job.get("descriptionPlain", "")
            )

            job_url = job.get("jobUrl", "") or f"https://jobs.ashbyhq.com/{token}/{job_id}"
            location = job.get("location", "")
            department = job.get("department", "")

            listing = make_listing(
                title=title,
                company=company,
                url=job_url,
                description=description,
                source="ashby",
                location=location,
                department=department,
            )
            if listing:
                listings.append(listing)

        logger.info(f"  ashby: {company} → {len(jobs)} jobs")

    logger.info(f"  ashby: {len(listings)} total listings")
    return listings


def fetch_workable(source: dict, watchlist: list, session: RateLimitedSession,
                   logger: logging.Logger) -> list[dict]:
    """Fetch from Workable public widget API for watched companies."""
    companies = [w for w in watchlist if w.get("ats") == "workable"]
    if not companies:
        return []

    listings = []

    for entry in companies:
        token = entry["board_token"]
        company = entry["company"]
        logger.info(f"  workable: fetching {company} (token={token})")

        # Get job list
        try:
            resp = session.get(
                f"https://apply.workable.com/api/v1/widget/accounts/{token}",
            )
            data = resp.json()
        except Exception as e:
            logger.warning(f"  workable: {company} job list failed: {e}")
            continue

        jobs = data.get("jobs", [])

        for job in jobs:
            shortcode = job.get("shortcode", "")
            title = job.get("title", "")
            shortlink = job.get("shortlink", "")
            location = job.get("location", "")
            department = job.get("department", "")

            # Fetch full description
            description = ""
            if shortcode:
                try:
                    detail_resp = session.get(
                        f"https://apply.workable.com/api/v1/widget/accounts/{token}/jobs/{shortcode}",
                    )
                    detail = detail_resp.json()
                    description = strip_html(detail.get("description", ""))
                except Exception as e:
                    logger.warning(f"  workable: {company} job {shortcode} detail failed: {e}")

            job_url = shortlink or f"https://apply.workable.com/{token}/j/{shortcode}/"

            listing = make_listing(
                title=title,
                company=company,
                url=job_url,
                description=description,
                source="workable",
                location=location,
                department=department,
            )
            if listing:
                listings.append(listing)

        logger.info(f"  workable: {company} → {len(jobs)} jobs")

    logger.info(f"  workable: {len(listings)} total listings")
    return listings


def fetch_websearch(source: dict, watchlist: list, session: RateLimitedSession,
                    logger: logging.Logger) -> list[dict]:
    """Best-effort scrape of careers pages for companies without ATS."""
    companies = [w for w in watchlist if w.get("ats") == "websearch"]
    if not companies:
        return []

    listings = []

    for entry in companies:
        company = entry["company"]
        url = entry.get("url", "")
        logger.info(f"  websearch: checking {company} ({url})")

        if not url:
            logger.warning(f"  websearch: {company} — no URL configured")
            continue

        try:
            resp = session.get(url)
            page_text = resp.text.lower()
        except Exception as e:
            logger.warning(f"  websearch: {company} fetch failed: {e}")
            continue

        # Look for career-related links or sections
        career_patterns = [
            r'href=["\']([^"\']*(?:career|job|position|opening|hiring)[^"\']*)["\']',
        ]
        career_urls = set()
        for pattern in career_patterns:
            for match in re.finditer(pattern, resp.text, re.IGNORECASE):
                found_url = match.group(1)
                if not found_url.startswith("http"):
                    found_url = url.rstrip("/") + "/" + found_url.lstrip("/")
                career_urls.add(found_url)

        if career_urls:
            logger.info(f"  websearch: {company} — found {len(career_urls)} career link(s)")
            for career_url in career_urls:
                listing = make_listing(
                    title=f"[Careers page] {company}",
                    company=company,
                    url=career_url,
                    description=f"Career page found at {career_url}. Visit to see open positions.",
                    source="websearch",
                    signal_level="low",
                )
                if listing:
                    listings.append(listing)
        else:
            # Check if page mentions hiring-related keywords at all
            if any(kw in page_text for kw in ["career", "hiring", "job", "position", "join us", "open role"]):
                logger.info(f"  websearch: {company} — hiring keywords found but no links extracted")
                listing = make_listing(
                    title=f"[Possible openings] {company}",
                    company=company,
                    url=url,
                    description=f"Hiring-related keywords found on {url}. Check manually.",
                    source="websearch",
                    signal_level="low",
                )
                if listing:
                    listings.append(listing)
            else:
                logger.info(f"  websearch: {company} — no career content found")

    logger.info(f"  websearch: {len(listings)} total listings")
    return listings


def fetch_gmail_alerts(source: dict, watchlist: list, session: RateLimitedSession,
                       logger: logging.Logger) -> list[dict]:
    """Fetch LinkedIn job alerts from Gmail API.

    Requires one-time OAuth setup:
      1. Create Google Cloud project → enable Gmail API
      2. Create OAuth credentials (desktop app) → download credentials.json
      3. Set GMAIL_CREDENTIALS_PATH env var
      4. First run opens browser for consent → saves token to state/gmail_token.json
    """
    credentials_path = _expand_env(source.get("credentials_path", ""))
    token_path = source.get("token_path", "./state/gmail_token.json")
    token_path = ROOT / token_path.lstrip("./")
    sender = source.get("sender", "jobs-noreply@linkedin.com")
    max_age_hours = source.get("max_age_hours", 24)

    if not credentials_path:
        logger.warning("  gmail_alerts: GMAIL_CREDENTIALS_PATH not set — skipping")
        return []

    if not Path(credentials_path).exists():
        logger.warning(f"  gmail_alerts: credentials file not found at {credentials_path} — skipping")
        return []

    # Import Gmail dependencies (optional — only needed if this adapter runs)
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request as GoogleRequest
        from googleapiclient.discovery import build
    except ImportError:
        logger.warning(
            "  gmail_alerts: missing dependencies — install with:\n"
            "    pip install google-api-python-client google-auth-oauthlib"
        )
        return []

    SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

    # Authenticate
    creds = None
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(GoogleRequest())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())
        logger.info("  gmail_alerts: token saved/refreshed")

    service = build("gmail", "v1", credentials=creds)

    # Search for LinkedIn alert emails
    query = f"from:{sender} newer_than:{max_age_hours}h"
    logger.info(f"  gmail_alerts: searching '{query}'")

    try:
        results = service.users().messages().list(userId="me", q=query, maxResults=50).execute()
    except Exception as e:
        logger.warning(f"  gmail_alerts: search failed: {e}")
        return []

    messages = results.get("messages", [])
    if not messages:
        logger.info("  gmail_alerts: no matching emails found")
        return []

    logger.info(f"  gmail_alerts: found {len(messages)} email(s)")
    listings = []

    for msg_info in messages:
        try:
            msg = service.users().messages().get(
                userId="me", id=msg_info["id"], format="full"
            ).execute()
        except Exception as e:
            logger.warning(f"  gmail_alerts: failed to read message {msg_info['id']}: {e}")
            continue

        # Extract HTML body
        html_body = _extract_gmail_html(msg)
        if not html_body:
            continue

        # Parse job cards from LinkedIn alert HTML
        cards = _parse_linkedin_alert(html_body)
        for card in cards:
            listing = make_listing(
                title=card["title"],
                company=card["company"],
                url=card["url"],
                description=card.get("snippet", ""),
                source="linkedin_alert",
                location=card.get("location", ""),
                signal_level="low",
            )
            if listing:
                listings.append(listing)

    logger.info(f"  gmail_alerts: {len(listings)} job cards extracted from {len(messages)} email(s)")
    return listings


def _extract_gmail_html(msg: dict) -> str:
    """Extract HTML body from a Gmail API message."""
    import base64

    payload = msg.get("payload", {})

    # Check if the body is directly in the payload
    if payload.get("mimeType") == "text/html":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    # Check parts
    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/html":
            data = part.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
        # Nested multipart
        for subpart in part.get("parts", []):
            if subpart.get("mimeType") == "text/html":
                data = subpart.get("body", {}).get("data", "")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    return ""


def fetch_jobspy(source: dict, watchlist: list, session: RateLimitedSession,
                 logger: logging.Logger) -> list[dict]:
    """Fetch from major job boards via python-jobspy (Indeed, Google Jobs, etc.)."""
    try:
        from jobspy import scrape_jobs
    except ImportError:
        logger.warning(
            "  jobspy: missing dependency — install with:\n"
            "    pip install python-jobspy"
        )
        return []

    params = source.get("params", {})
    search_terms = params.get("search_terms", ["product manager", "customer success manager"])
    site_names = params.get("sites", ["google", "indeed"])
    results_wanted = params.get("results_wanted", 30)
    hours_old = params.get("hours_old", 24)
    is_remote = params.get("is_remote", True)
    country = params.get("country", "USA")

    listings = []
    for term in search_terms:
        logger.info(f"  jobspy: searching '{term}' on {site_names}")
        try:
            jobs_df = scrape_jobs(
                site_name=site_names,
                search_term=term,
                results_wanted=results_wanted,
                hours_old=hours_old,
                is_remote=is_remote,
                country_indeed=country,
            )
        except Exception as e:
            logger.warning(f"  jobspy: search '{term}' failed: {e}")
            continue

        if jobs_df is None or jobs_df.empty:
            logger.info(f"  jobspy: no results for '{term}'")
            continue

        for _, row in jobs_df.iterrows():
            job_url = str(row.get("job_url", "")) or ""
            if not job_url or job_url == "nan":
                continue

            title = str(row.get("title", "")) or ""
            company = str(row.get("company", "")) or ""
            description = str(row.get("description", "")) or ""
            location = str(row.get("location", "")) or ""
            site = str(row.get("site", "")) or ""

            # Build extra fields from available columns
            extra = {"location": location}
            if site and site != "nan":
                extra["board"] = site

            # Salary info if available
            min_salary = row.get("min_amount")
            max_salary = row.get("max_amount")
            if min_salary and str(min_salary) != "nan":
                extra["salary_min"] = float(min_salary)
            if max_salary and str(max_salary) != "nan":
                extra["salary_max"] = float(max_salary)

            listing = make_listing(
                title=title,
                company=company if company != "nan" else "",
                url=job_url,
                description=description if description != "nan" else "",
                source=f"jobspy_{site}" if site and site != "nan" else "jobspy",
                **extra,
            )
            if listing:
                listings.append(listing)

    logger.info(f"  jobspy: {len(listings)} listings from {len(search_terms)} search terms")
    return listings


def _parse_linkedin_alert(html_body: str) -> list[dict]:
    """Parse job cards from LinkedIn job alert email HTML.

    LinkedIn wraps each job card in an <a> tag containing nested tables.
    All card info (title, company, location, salary) is text inside the <a>.
    Text content follows: "Title Company · Location Salary".
    """
    cards = []
    seen_urls = set()

    # Match full <a href="...jobs/view/...">...inner HTML...</a>
    anchor_pattern = re.compile(
        r'<a\s+href="([^"]*(?:/jobs/view/|/comm/jobs/view/)[^"]*)"[^>]*>'
        r'(.*?)</a>',
        re.DOTALL | re.IGNORECASE,
    )

    for m in anchor_pattern.finditer(html_body):
        url = m.group(1)
        inner_html = m.group(2)

        # Normalize URL: strip tracking params
        clean_url = re.sub(r'\?.*$', '', url)
        if clean_url in seen_urls:
            continue
        seen_urls.add(clean_url)

        # Strip HTML tags, decode entities, normalize whitespace
        text = re.sub(r'<[^>]+>', ' ', inner_html)
        text = text.replace('&middot;', '·').replace('&amp;', '&')
        text = ' '.join(text.split()).strip()

        if not text:
            continue

        # Parse: "Title Company · Location Salary"
        # Split on · (middot) — first part has title + company, rest has location + salary
        parts = text.split('·', 1)
        title_company = parts[0].strip()
        location_salary = parts[1].strip() if len(parts) > 1 else ""

        # Title is typically longer; company is the last 1-2 words.
        # LinkedIn puts them as separate text nodes but after stripping tags
        # they merge. Heuristic: split on double-space or use the last
        # short word group as company.
        # Better heuristic: look for the split in the inner HTML — title and
        # company are in separate table cells.
        title = ""
        company = ""
        # Find distinct text blocks in the inner HTML (each <td> text)
        text_blocks = re.findall(r'>([^<]{2,})<', inner_html)
        text_blocks = [b.strip() for b in text_blocks if b.strip()]
        if text_blocks:
            title = text_blocks[0]
            if len(text_blocks) > 1:
                # Company block may include " · Location" — split on middot
                raw_company = text_blocks[1]
                raw_company = raw_company.replace('&middot;', '·')
                company = raw_company.split('·')[0].strip()

        # Extract location (before any salary pattern)
        location = ""
        if location_salary:
            # Remove salary pattern like "$100K-$225K / year"
            loc = re.sub(r'\$[\d,KkMm].*$', '', location_salary).strip()
            location = loc if loc else location_salary

        cards.append({
            "title": title or "[LinkedIn job]",
            "company": company or "[Unknown company]",
            "url": clean_url,
            "snippet": text,
            "location": location,
        })

    return cards

# ---------------------------------------------------------------------------
# Adapter dispatch
# ---------------------------------------------------------------------------

ADAPTERS = {
    "himalayas": fetch_himalayas,
    "jobicy": fetch_jobicy,
    "greenhouse": fetch_greenhouse,
    "ashby": fetch_ashby,
    "workable": fetch_workable,
    "websearch": fetch_websearch,
    "gmail_alerts": fetch_gmail_alerts,
    "jobspy": fetch_jobspy,
}


def run_adapters(sources_cfg: dict, watchlist: list, session: RateLimitedSession,
                 logger: logging.Logger, source_filter: str | None = None) -> list[dict]:
    """Run all enabled adapters and collect listings."""
    all_listings = []

    for source in sources_cfg.get("sources", []):
        source_id = source["id"]
        source_type = source["type"]

        if not source.get("enabled", True):
            logger.debug(f"Skipping disabled source: {source_id}")
            continue

        if source_filter and source_id != source_filter:
            continue

        adapter = ADAPTERS.get(source_type)
        if not adapter:
            logger.warning(f"No adapter for source type '{source_type}' (source: {source_id})")
            continue

        logger.info(f"Fetching: {source_id} (type={source_type})")
        try:
            listings = adapter(source, watchlist, session, logger)
            all_listings.extend(listings)
        except Exception as e:
            logger.error(f"Source {source_id} failed: {e}")
            # Continue with other sources — don't let one break everything

    return all_listings


def dedup_listings(listings: list[dict], logger: logging.Logger) -> list[dict]:
    """Deduplicate listings by URL."""
    seen_urls = set()
    unique = []
    for listing in listings:
        url = listing["url"]
        if url not in seen_urls:
            seen_urls.add(url)
            unique.append(listing)
    dupes = len(listings) - len(unique)
    if dupes:
        logger.info(f"Deduplication: {dupes} duplicate(s) removed, {len(unique)} unique listings")
    return unique

# ---------------------------------------------------------------------------
# Verify mode
# ---------------------------------------------------------------------------

def verify_sources(sources_cfg: dict, watchlist: list, session: RateLimitedSession,
                   logger: logging.Logger):
    """Test each enabled source — check connectivity and print job counts."""
    print("\n=== Source Verification ===\n")

    for source in sources_cfg.get("sources", []):
        source_id = source["id"]
        if not source.get("enabled", True):
            print(f"  {source_id}: DISABLED (skipped)")
            continue

        source_type = source["type"]

        # Test connectivity per type
        try:
            if source_type == "himalayas":
                resp = session.get(source["base_url"], params={"limit": 1})
                jobs = resp.json().get("jobs", [])
                print(f"  ✓ {source_id}: OK (API reachable, sample: {len(jobs)} job(s))")

            elif source_type == "jobicy":
                resp = session.get(source["base_url"], params={"count": 1})
                jobs = resp.json().get("jobs", [])
                print(f"  ✓ {source_id}: OK (API reachable, sample: {len(jobs)} job(s))")

            elif source_type == "greenhouse":
                companies = [w for w in watchlist if w.get("ats") == "greenhouse"]
                for entry in companies:
                    token = entry["board_token"]
                    resp = session.get(
                        f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs"
                    )
                    jobs = resp.json().get("jobs", [])
                    print(f"  ✓ {source_id}/{entry['company']}: OK ({len(jobs)} jobs)")

            elif source_type == "ashby":
                companies = [w for w in watchlist if w.get("ats") == "ashby"]
                for entry in companies:
                    token = entry["board_token"]
                    resp = session.get(
                        f"https://api.ashbyhq.com/posting-api/job-board/{token}",
                    )
                    jobs = resp.json().get("jobs", [])
                    print(f"  ✓ {source_id}/{entry['company']}: OK ({len(jobs)} jobs)")

            elif source_type == "workable":
                companies = [w for w in watchlist if w.get("ats") == "workable"]
                for entry in companies:
                    token = entry["board_token"]
                    resp = session.get(
                        f"https://apply.workable.com/api/v1/widget/accounts/{token}"
                    )
                    jobs = resp.json().get("jobs", [])
                    print(f"  ✓ {source_id}/{entry['company']}: OK ({len(jobs)} jobs)")

            elif source_type == "websearch":
                companies = [w for w in watchlist if w.get("ats") == "websearch"]
                for entry in companies:
                    url = entry.get("url", "")
                    if url:
                        resp = session.get(url)
                        print(f"  ✓ {source_id}/{entry['company']}: OK (HTTP {resp.status_code})")
                    else:
                        print(f"  ? {source_id}/{entry['company']}: no URL configured")

            elif source_type == "gmail_alerts":
                creds_path = _expand_env(source.get("credentials_path", ""))
                if not creds_path:
                    print(f"  ⚠ {source_id}: GMAIL_CREDENTIALS_PATH not set")
                elif not Path(creds_path).exists():
                    print(f"  ⚠ {source_id}: credentials file not found at {creds_path}")
                else:
                    print(f"  ✓ {source_id}: credentials file found (OAuth flow not tested)")

            else:
                print(f"  ? {source_id}: unknown type '{source_type}'")

        except Exception as e:
            print(f"  ✗ {source_id}: FAILED — {e}")

    print()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Perian data collector — fetch job listings from boards and ATS watchlists"
    )
    parser.add_argument("--source", help="Run single source by ID")
    parser.add_argument("--dry-run", action="store_true", help="Fetch but don't write JSONL")
    parser.add_argument("--verify", action="store_true", help="Test source connectivity")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    args = parser.parse_args()

    logger = setup_logging(args.verbose)

    # Load config
    sources_cfg = load_sources()
    watchlist = load_watchlist()
    defaults = sources_cfg.get("defaults", {})

    logger.info(f"Loaded {len(sources_cfg.get('sources', []))} sources, {len(watchlist)} watchlist companies")

    # Build session with defaults
    session = RateLimitedSession(
        rate_limit=defaults.get("rate_limit_seconds", 2),
        timeout=defaults.get("timeout_seconds", 30),
        user_agent=defaults.get("user_agent", "Perian/1.0"),
    )

    # Verify mode
    if args.verify:
        verify_sources(sources_cfg, watchlist, session, logger)
        return

    # Collect
    logger.info("=== Collection starting ===")
    start_time = time.time()
    listings = run_adapters(sources_cfg, watchlist, session, logger, args.source)
    listings = dedup_listings(listings, logger)
    elapsed = time.time() - start_time

    logger.info(f"Collection complete: {len(listings)} unique listings in {elapsed:.1f}s")

    if args.dry_run:
        logger.info("[DRY RUN] Would write the following:")
        for i, l in enumerate(listings[:10], 1):
            logger.info(f"  {i}. {l['title']} @ {l['company']} ({l['source']})")
        if len(listings) > 10:
            logger.info(f"  ... and {len(listings) - 10} more")
        return

    # Write
    date_str = datetime.now().strftime("%Y-%m-%d")
    write_jsonl(listings, date_str, logger)

    # Cleanup old files
    cleanup_old_jsonl(days=14, logger=logger)

    # Write collection summary for Shadowfax timing
    summary_path = Path(__file__).resolve().parent.parent / "state" / "collect-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    collect_summary = {
        "date": date_str,
        "duration_s": round(elapsed, 1),
        "listings_collected": len(listings),
    }
    with open(summary_path, "w") as f:
        json.dump(collect_summary, f, indent=2)

    logger.info("=== Collection done ===")


if __name__ == "__main__":
    main()
