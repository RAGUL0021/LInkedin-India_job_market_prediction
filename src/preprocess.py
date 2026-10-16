"""Cleaning + feature engineering for the LinkedIn job-market dataset (India).

Works with BOTH:
  * the original raw scrape columns (job, location, no_of_employ, ...)
  * the friendly template columns (job_title, company_size, posted_hours_ago, ...)
"""
import re
import numpy as np
import pandas as pd

CAT_COLS = ["role_family", "city", "work_type", "employment_type", "seniority",
            "company_size", "industry", "company"]
NUM_COLS = ["posted_hours", "log_alumni", "log_followers", "desc_len", "n_skills",
            "hiring_person_listed", "title_len"]
TEXT_COL = "title_text"
FEATURE_COLS = CAT_COLS + NUM_COLS + [TEXT_COL]

TEMPLATE_COLS = ["job_title", "company_name", "location", "work_type", "employment_type",
                 "seniority", "company_size", "industry", "posted_hours_ago",
                 "alumni_count", "company_followers", "hiring_person_listed", "description"]

ROLE_RULES = [
    ("Data Engineer", r"data engineer|etl|big data|databricks|snowflake|spark|hadoop"),
    ("Data Scientist / ML / AI", r"data scien|machine learning|\bml\b|\bai\b|nlp|deep learning|genai|llm"),
    ("Data / Business Analyst", r"data analyst|business analyst|\bbi\b|power ?bi|tableau|analytics|analyst"),
    ("QA / Testing", r"test|qa\b|quality assurance|sdet|automation"),
    ("Java Developer", r"java(?!script)|spring|j2ee"),
    ("Python Developer", r"python|django|flask|fastapi"),
    (".NET Developer", r"\.net|dotnet|c#|asp\.net"),
    ("Frontend / Full Stack", r"react|angular|vue|front.?end|full.?stack|javascript|node|mern|mean\b|ui developer"),
    ("DevOps / Cloud", r"devops|cloud|aws|azure|gcp|sre|kubernetes|infrastructure|platform engineer"),
    ("Salesforce / SAP / ERP", r"salesforce|sap\b|servicenow|workday|oracle|dynamics|sharepoint|tibco|mulesoft"),
    ("Mobile", r"android|ios|mobile|flutter|swift|kotlin|react native"),
    ("Management / Lead", r"manager|director|architect|head of|scrum|product owner|project"),
    ("Software Engineer (General)", r"software|developer|engineer|programmer"),
]
SKILL_WORDS = ["python", "java", "sql", "aws", "azure", "gcp", "spark", "kafka", "docker", "kubernetes",
               "react", "angular", "node", "machine learning", "tableau", "power bi", "excel", "selenium",
               "salesforce", "etl", "hadoop", "scala", "javascript", "typescript", "c++", "linux", "git",
               "agile", "microservices", "rest", "api", "snowflake", "databricks", "terraform", "jenkins"]


def _isnull(x):
    return x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NA


def role_family(title: str) -> str:
    t = str(title).lower()
    for name, pat in ROLE_RULES:
        if re.search(pat, t):
            return name
    return "Other"


def parse_time_to_hours(s):
    """'8 hours' -> 8, '2 days' -> 48, '26 minutes' -> 0.43, '1 week' -> 168."""
    if _isnull(s):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(minute|hour|day|week|month)", str(s).lower())
    if not m:
        try:
            return float(s)
        except ValueError:
            return np.nan
    n, unit = float(m.group(1)), m.group(2)
    return n * {"minute": 1 / 60, "hour": 1, "day": 24, "week": 168, "month": 720}[unit]


def parse_int(s):
    if _isnull(s):
        return np.nan
    m = re.search(r"\d[\d,]*", str(s))
    return float(m.group(0).replace(",", "")) if m else np.nan


def parse_city(loc):
    if _isnull(loc):
        return "Unknown"
    s = str(loc)
    if s.strip().lower() == "india":
        return "India (unspecified)"
    first = s.split(",")[0].strip()
    first = re.sub(r"\s*(district|area|metropolitan region)$", "", first, flags=re.I)
    first = re.sub(r"^greater\s+", "", first, flags=re.I)
    return {"New Delhi": "Delhi", "Bangalore": "Bengaluru", "Gurgaon": "Gurugram"}.get(first, first)


def parse_company_size_industry(s):
    if _isnull(s):
        return "Unknown", "Unknown"
    parts = [p.strip() for p in str(s).split("·")]
    size = parts[0].replace("employees", "").strip() if parts else "Unknown"
    ind = parts[1] if len(parts) > 1 else "Unknown"
    return size or "Unknown", ind or "Unknown"


def parse_employment(s):
    if _isnull(s):
        return "Unknown", "Unknown"
    parts = [p.strip() for p in str(s).split("·")]
    return parts[0] or "Unknown", (parts[1] if len(parts) > 1 else "Not specified")


def seniority_from_title(title):
    t = str(title).lower()
    if re.search(r"intern|trainee|fresher|junior|\bjr\b|entry", t):
        return "Entry level"
    if re.search(r"lead|principal|staff|architect|manager|head|\bsr\b|senior", t):
        return "Mid-Senior level"
    return None


def _skill_count(text):
    t = str(text).lower()
    return sum(1 for w in SKILL_WORDS if w in t)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw OR template columns into one clean, human-readable frame."""
    d = pd.DataFrame(index=df.index)
    raw = "job" in df.columns and "job_title" not in df.columns
    if raw:
        d["job_title"] = df["job"]
        d["company_name"] = df.get("company_name")
        d["location"] = df.get("location")
        d["work_type"] = df.get("work_type")
        emp = df["full_time_remote"].apply(parse_employment)
        d["employment_type"] = [e[0] for e in emp]
        d["seniority"] = [e[1] for e in emp]
        si = df["no_of_employ"].apply(parse_company_size_industry)
        d["company_size"] = [x[0] for x in si]
        d["industry"] = [x[1] for x in si]
        d["posted_hours_ago"] = df["posted_day_ago"].apply(parse_time_to_hours)
        d["alumni_count"] = df["alumni"].apply(parse_int)
        d["company_followers"] = df["linkedin_followers"].apply(parse_int)
        d["hiring_person_listed"] = df["Hiring_person"].notna().astype(int)
        d["description"] = df["job_details"]
    else:
        for c in TEMPLATE_COLS:
            d[c] = df[c] if c in df.columns else np.nan
        d["hiring_person_listed"] = pd.to_numeric(d["hiring_person_listed"], errors="coerce")
    return d


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = standardize(df)
    f = pd.DataFrame(index=d.index)
    title = d["job_title"].fillna("").astype(str)
    f["title_text"] = title.str.lower().str.replace(r"[^a-z0-9#+. ]", " ", regex=True)
    f["role_family"] = title.apply(role_family)
    f["city"] = d["location"].apply(parse_city)
    f["work_type"] = (d["work_type"].fillna("Unknown").astype(str).str.strip().str.title()
                      .replace({"On-Site": "On-site"}))
    f["employment_type"] = d["employment_type"].fillna("Unknown").astype(str)
    sen = d["seniority"].fillna("").astype(str)
    guess = title.apply(seniority_from_title)
    f["seniority"] = [s if s and s != "Unknown" else (g or "Not specified") for s, g in zip(sen, guess)]
    f["company_size"] = d["company_size"].fillna("Unknown").astype(str).str.replace(" employees", "", regex=False)
    f["industry"] = d["industry"].fillna("Unknown").astype(str)
    f["company"] = d["company_name"].fillna("Unknown").astype(str)
    f["posted_hours"] = pd.to_numeric(d["posted_hours_ago"], errors="coerce")
    f["log_alumni"] = np.log1p(pd.to_numeric(d["alumni_count"], errors="coerce"))
    f["log_followers"] = np.log1p(pd.to_numeric(d["company_followers"], errors="coerce"))
    desc = d["description"].fillna("").astype(str)
    f["desc_len"] = desc.str.len().astype(float).replace(0, np.nan)
    f["n_skills"] = desc.apply(_skill_count).astype(float).where(desc.str.len() > 0)
    f["hiring_person_listed"] = pd.to_numeric(d["hiring_person_listed"], errors="coerce").fillna(0)
    f["title_len"] = title.str.len()
    return f[FEATURE_COLS]


def load_training_data(path: str):
    """Return (features, target, clean_df). Target = number of applicants.

    Rows where LinkedIn showed no applicant count contain the words
    'hours' / 'days' / 'minutes' instead of a number -> dropped for training.
    LinkedIn caps the visible count at 200 (right-censored target).
    """
    raw = pd.read_csv(path)
    raw = raw.drop(columns=[c for c in raw.columns if c.startswith("Column")], errors="ignore")
    raw["applicants"] = pd.to_numeric(raw["no_of_application"].astype(str).str.replace(",", ""), errors="coerce")
    raw = raw.dropna(subset=["applicants", "job"]).drop_duplicates(subset=["job_ID"]).reset_index(drop=True)
    X = build_features(raw)
    y = raw["applicants"].astype(float)
    clean = standardize(raw)
    clean["applicants"] = y.values
    clean["city"] = X["city"].values
    clean["role_family"] = X["role_family"].values
    return X, y, clean
