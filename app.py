import re

import streamlit as st
import pandas as pd
import plotly.express as px

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Saudi Tech Job Market",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Executive CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* ── Spacing ── */
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

/* ── Page header ── */
.exec-title {
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0 0 8px 0;
}
.exec-meta {
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    opacity: 0.38;
    margin: 0;
}

/* ── KPI blocks ── */
.kpi-block {
    padding: 18px 4px 14px 4px;
    border-top: 2px solid rgba(255,255,255,0.12);
}
.kpi-number {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    font-variant-numeric: tabular-nums;
    line-height: 1;
}
.kpi-label {
    font-size: 0.67rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    opacity: 0.4;
    margin-top: 7px;
}

/* ── Insight cards ── */
.insight-block {
    padding: 16px 18px;
    border-left: 2px solid rgba(255,255,255,0.16);
    background: rgba(255,255,255,0.03);
    border-radius: 0 6px 6px 0;
    min-height: 90px;
}
.insight-tag {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    opacity: 0.38;
    margin-bottom: 8px;
}
.insight-val {
    font-size: 1.2rem;
    font-weight: 600;
    line-height: 1.2;
}
.insight-hint {
    font-size: 0.72rem;
    opacity: 0.4;
    margin-top: 5px;
}

/* ── Section labels ── */
.sec-label {
    display: block;
    font-size: 0.63rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    opacity: 0.38;
    margin-bottom: 0.6rem;
}

/* ── Result badges ── */
.badge {
    display: inline-block;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 9px;
    border-radius: 3px;
    margin-bottom: 12px;
}
.badge-pub {
    background: rgba(148,163,184,0.1);
    color: #94a3b8;
    border: 1px solid rgba(148,163,184,0.2);
}
.badge-live {
    background: rgba(99,102,241,0.1);
    color: #818cf8;
    border: 1px solid rgba(99,102,241,0.2);
}

/* ── Chart notes ── */
.chart-note {
    font-size: 0.74rem;
    opacity: 0.4;
    line-height: 1.55;
    margin: -4px 0 18px 0;
}

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_jobs() -> pd.DataFrame:
    return pd.read_csv("jobs_sa_cleaned.csv")


def normalize_result_df(
    file_path: str, label_name: str = None, count_name: str = None
) -> pd.DataFrame:
    """
    Load a precomputed result CSV and return a clean two-column DataFrame
    (Label, Count).  label_name / count_name are tried first; if absent the
    function falls back to a known-column priority list.
    """
    df = pd.read_csv(file_path)
    label_priority = ([label_name] if label_name else []) + [
        "Skill", "job_city", "job_title_std", "role_group", "Label"
    ]
    count_priority = ([count_name] if count_name else []) + [
        "Count", "count", "Value", "value"
    ]
    label_col = next((c for c in label_priority if c in df.columns), df.columns[0])
    count_col = next((c for c in count_priority if c in df.columns), df.columns[-1])
    result = df[[label_col, count_col]].copy()
    result.columns = ["Label", "Count"]
    return result.dropna().reset_index(drop=True)


CHART_TYPES = ["Horizontal Bar", "Bar", "Pie", "Treemap"]
COLOR_SCALE = "Blues"
QUAL_COLORS = px.colors.qualitative.Pastel
TEMPLATE    = "plotly_dark"

ROLE_KEYWORDS: dict[str, list[str]] = {
    "AI/ML Engineer":    ["ai engineer", "machine learning", "ml engineer", "deep learning",
                          "llm", "nlp engineer", "computer vision", "ai/ml"],
    "Data Engineer":     ["data engineer", "etl", "data pipeline", "data infrastructure"],
    "Software Engineer": ["software engineer", "software developer", "full stack", "fullstack"],
    "Backend Developer": ["backend developer", "back-end developer", "backend engineer",
                          "back end developer"],
    "Data Scientist":    ["data scientist"],
    "Data Analyst":      ["data analyst", "business analyst", "bi analyst", "analytics engineer"],
}


def classify_role(title: str) -> str:
    tl = title.lower()
    for role, keywords in ROLE_KEYWORDS.items():
        if any(kw in tl for kw in keywords):
            return role
    return "Other"


@st.cache_data
def compute_live_skills(clean_desc: pd.Series) -> pd.DataFrame:
    """Re-extract skill counts from a filtered description Series using the published taxonomy."""
    if clean_desc.empty:
        return pd.DataFrame(columns=["Label", "Count"])
    desc = clean_desc.fillna("").str.lower()
    rows = [
        {
            "Label": skill,
            "Count": int(
                desc.str.contains(
                    r"\b" + re.escape(skill.lower()) + r"\b", regex=True
                ).sum()
            ),
        }
        for skill in skills_df["Label"]
    ]
    result = pd.DataFrame(rows)
    return (
        result[result["Count"] > 0]
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )


@st.cache_data
def compute_live_roles(job_titles: pd.Series) -> pd.DataFrame:
    """Classify raw job titles into role groups and return frequency counts."""
    if job_titles.empty:
        return pd.DataFrame(columns=["Label", "Count"])
    classified = job_titles.dropna().map(classify_role)
    result = classified.value_counts().reset_index()
    result.columns = ["Label", "Count"]
    return result[result["Count"] > 0].reset_index(drop=True)


def safe_slider(label: str, min_n: int, available: int, default_n: int, key: str) -> int:
    """
    Show st.slider only when a valid range exists (available > min_n).
    - available <= 0  : data is empty — return default_n so other charts still render
    - available <= min_n : too few items for a range — return available directly, no widget
    - available >  min_n : show slider with min_value=min_n, max_value=available
    Guarantees min_value < max_value is always satisfied.
    """
    if available <= 0:
        return default_n
    if available <= min_n:
        return available
    return st.slider(label, min_value=min_n, max_value=available,
                     value=min(default_n, available), key=key)


def plot_result(
    df: pd.DataFrame,
    label_col: str,
    count_col: str,
    title: str,
    chart_type: str = "Horizontal Bar",
    top_n: int = 10,
    height: int = 420,
) -> None:
    """Render a Plotly chart for any result DataFrame. Shows st.warning if empty."""
    if df is None or df.empty:
        st.warning(f"No data available for: {title}")
        return

    data   = df.head(top_n).copy()
    common = dict(template=TEMPLATE, title=title, height=height)

    if chart_type == "Horizontal Bar":
        fig = px.bar(
            data.sort_values(count_col),
            x=count_col, y=label_col, orientation="h",
            color=count_col, color_continuous_scale=COLOR_SCALE,
            **common,
        )
        fig.update_layout(
            yaxis_title=None, xaxis_title="Job Postings",
            coloraxis_showscale=False,
        )
        fig.update_traces(hovertemplate="%{y}: %{x} postings<extra></extra>")

    elif chart_type == "Bar":
        fig = px.bar(
            data.sort_values(count_col, ascending=False),
            x=label_col, y=count_col,
            color=count_col, color_continuous_scale=COLOR_SCALE,
            **common,
        )
        fig.update_layout(
            xaxis_title=None, yaxis_title="Job Postings",
            coloraxis_showscale=False,
            xaxis_tickangle=-35,
        )
        fig.update_traces(hovertemplate="%{x}: %{y} postings<extra></extra>")

    elif chart_type == "Pie":
        fig = px.pie(
            data, names=label_col, values=count_col, hole=0.42,
            color_discrete_sequence=QUAL_COLORS,
            **common,
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="%{label}: %{value} postings (%{percent})<extra></extra>",
        )
        fig.update_layout(showlegend=False)

    elif chart_type == "Treemap":
        fig = px.treemap(
            data,
            path=[px.Constant("All"), label_col],
            values=count_col,
            color=count_col,
            color_continuous_scale=COLOR_SCALE,
            **common,
        )
        fig.update_traces(hovertemplate="%{label}: %{value}<extra></extra>")
        fig.update_layout(coloraxis_showscale=False)

    else:
        st.warning(f"Unknown chart type: {chart_type}")
        return

    fig.update_layout(margin=dict(l=10, r=10, t=46, b=16))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

jobs_df   = load_jobs()
skills_df = normalize_result_df("skills_results.csv",  "Skill",         "Count")
cities_df = normalize_result_df("cities_results.csv",  "job_city",      "count")
titles_df = normalize_result_df("titles_results.csv",  "job_title_std", "count")
roles_df  = normalize_result_df("roles_results.csv",   "role_group",    "count")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════

hdr_l, hdr_r = st.columns([5, 1])
with hdr_l:
    st.markdown(
        '<h1 class="exec-title">Saudi Tech &amp; Data Job Market</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="exec-meta">'
        'Executive Analytics Dashboard &nbsp;&middot;&nbsp; '
        'Data Science Level 6 &nbsp;&middot;&nbsp; '
        'Kingdom of Saudi Arabia &nbsp;&middot;&nbsp; 2024'
        '</p>',
        unsafe_allow_html=True,
    )
with hdr_r:
    st.markdown(
        f'<p style="text-align:right;font-size:0.72rem;opacity:0.35;margin-top:28px;">'
        f'{len(jobs_df):,} postings analysed'
        f'</p>',
        unsafe_allow_html=True,
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Executive Overview",
    "Skills Intelligence",
    "Roles & Titles",
    "Location Insights",
    "Interactive Analysis",
    "Data Explorer",
])

# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — EXECUTIVE OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── KPI row ───────────────────────────────────────────────────────────────
    n_jobs   = len(jobs_df)
    n_comp   = jobs_df["employer_name"].nunique() if "employer_name" in jobs_df.columns else 0
    n_ttls   = jobs_df["job_title"].nunique()     if "job_title"     in jobs_df.columns else 0
    n_cities = jobs_df["job_city"].nunique()      if "job_city"      in jobs_df.columns else 0

    k1, k2, k3, k4 = st.columns(4)
    for col, val, label in [
        (k1, f"{n_jobs:,}",   "Total Job Postings"),
        (k2, f"{n_comp:,}",   "Unique Companies"),
        (k3, f"{n_ttls:,}",   "Unique Job Titles"),
        (k4, f"{n_cities:,}", "Cities Covered"),
    ]:
        col.markdown(
            f'<div class="kpi-block">'
            f'<div class="kpi-number">{val}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── Key Insights ──────────────────────────────────────────────────────────
    st.markdown('<span class="sec-label">Key Insights</span>', unsafe_allow_html=True)

    top_skill       = skills_df.iloc[0]["Label"]      if not skills_df.empty else "N/A"
    top_skill_count = int(skills_df.iloc[0]["Count"]) if not skills_df.empty else 0
    top_city_name   = cities_df.iloc[0]["Label"]      if not cities_df.empty else "N/A"
    top_city_count  = int(cities_df.iloc[0]["Count"]) if not cities_df.empty else 0
    top_role        = roles_df.iloc[0]["Label"]       if not roles_df.empty else "N/A"
    top_role_count  = int(roles_df.iloc[0]["Count"])  if not roles_df.empty else 0
    top5_share      = (
        round(skills_df.head(5)["Count"].sum() / skills_df["Count"].sum() * 100)
        if not skills_df.empty else 0
    )

    i1, i2, i3, i4 = st.columns(4)
    for col, tag, val, hint in [
        (i1, "Most Demanded Skill",   top_skill.title(),  f"{top_skill_count:,} postings"),
        (i2, "Top Hiring City",       top_city_name,      f"{top_city_count:,} postings"),
        (i3, "Leading Role Group",    top_role,           f"{top_role_count:,} postings"),
        (i4, "Top-5 Skills Coverage", f"{top5_share}%",   f"of all skill mentions — {len(skills_df)} skills tracked"),
    ]:
        col.markdown(
            f'<div class="insight-block">'
            f'<div class="insight-tag">{tag}</div>'
            f'<div class="insight-val">{val}</div>'
            f'<div class="insight-hint">{hint}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── 2 x 2 chart grid ─────────────────────────────────────────────────────
    st.markdown('<span class="sec-label">Market Snapshot — Published Analysis Results</span>',
                unsafe_allow_html=True)
    st.caption(
        "Based on the full 1,020-posting dataset. "
        "Visit the Interactive Analysis tab to explore filtered subsets."
    )

    row1a, row1b = st.columns(2)
    with row1a:
        plot_result(skills_df, "Label", "Count", "Top 10 Skills", "Horizontal Bar", 10, height=340)
        st.markdown(
            '<p class="chart-note">Python and Machine Learning lead by a wide margin, '
            'confirming the Kingdom\'s AI-first talent demand.</p>',
            unsafe_allow_html=True,
        )
    with row1b:
        plot_result(roles_df, "Label", "Count", "Role Distribution", "Pie", len(roles_df), height=340)
        st.markdown(
            '<p class="chart-note">AI/ML and Data Engineering together represent '
            'the largest share of the market.</p>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    row2a, row2b = st.columns(2)
    with row2a:
        plot_result(cities_df, "Label", "Count", "Top Cities", "Horizontal Bar", 10, height=340)
        st.markdown(
            '<p class="chart-note">Riyadh concentrates the majority of postings, '
            'driven by Vision 2030 digital investment.</p>',
            unsafe_allow_html=True,
        )
    with row2b:
        plot_result(titles_df, "Label", "Count", "Top Standardized Titles", "Horizontal Bar", 10, height=340)
        st.markdown(
            '<p class="chart-note">AI Engineer and Backend Developer lead '
            'standardized title demand across the sector.</p>',
            unsafe_allow_html=True,
        )

# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — SKILLS INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Skills Intelligence")
    st.markdown('<span class="badge badge-pub">Published Results — Full Dataset</span>',
                unsafe_allow_html=True)
    st.caption(
        "Frequency counts derived from keyword matching against normalized job descriptions. "
        "Each value represents distinct postings mentioning the skill at least once."
    )

    ctrl1, ctrl2, _ = st.columns([2, 3, 2])
    with ctrl1:
        sk_chart = st.selectbox("Chart type", CHART_TYPES, key="sk_chart")
    with ctrl2:
        sk_n = safe_slider("Top N", 5, len(skills_df), 15, "sk_n")

    plot_result(skills_df, "Label", "Count",
                f"Top {sk_n} Most Demanded Technical Skills", sk_chart, sk_n)
    st.markdown(
        '<p class="chart-note">'
        'Python leads across all role types. Machine Learning and SQL follow as foundational competencies. '
        'Cloud platforms and DevOps tooling appear mid-tier, reflecting growing production-deployment demand.'
        '</p>',
        unsafe_allow_html=True,
    )

    st.divider()

    col_tbl, col_stats = st.columns([3, 1])
    with col_tbl:
        st.markdown('<span class="sec-label">Full Skill Inventory</span>', unsafe_allow_html=True)
        display_skills = (
            skills_df
            .rename(columns={"Label": "Skill", "Count": "Postings"})
            .assign(
                Share=lambda d: (
                    d["Postings"] / d["Postings"].sum() * 100
                ).round(1).astype(str) + "%"
            )
        )
        st.dataframe(display_skills, use_container_width=True, hide_index=True, height=340)

    with col_stats:
        st.markdown('<span class="sec-label">Summary</span>', unsafe_allow_html=True)
        top3_share = round(
            skills_df.head(3)["Count"].sum() / skills_df["Count"].sum() * 100
        )
        st.metric("Skills Tracked",  str(len(skills_df)))
        st.metric("Top-3 Share",     f"{top3_share}%")
        st.metric("Peak Skill",      skills_df.iloc[0]["Label"].title() if not skills_df.empty else "N/A")
        st.metric("Peak Mentions",   f"{int(skills_df.iloc[0]['Count']):,}" if not skills_df.empty else "N/A")

# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — ROLES & TITLES
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Roles & Titles")
    st.markdown('<span class="badge badge-pub">Published Results — Full Dataset</span>',
                unsafe_allow_html=True)

    # Roles
    st.markdown('<span class="sec-label">Role Group Distribution</span>', unsafe_allow_html=True)
    st.caption("Titles mapped to seven role groups via standardization dictionary.")

    ro_ctrl1, _, _ = st.columns([2, 2, 3])
    with ro_ctrl1:
        ro_chart = st.selectbox("Chart type", CHART_TYPES, key="ro_chart")

    col_bar, col_donut = st.columns([3, 2])
    with col_bar:
        plot_result(roles_df, "Label", "Count",
                    "Role Groups by Volume", ro_chart, len(roles_df))
    with col_donut:
        if ro_chart != "Pie":
            plot_result(roles_df, "Label", "Count",
                        "Market Share by Role", "Pie", len(roles_df))

    st.markdown(
        '<p class="chart-note">'
        'AI/ML Engineer and Data Engineer are nearly equal in volume. '
        'Together with Software and Backend roles, these four groups account for over 80% of all postings.'
        '</p>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Titles
    st.markdown('<span class="sec-label">Standardized Job Titles</span>', unsafe_allow_html=True)
    st.caption("Raw titles normalized to canonical forms — location suffixes and seniority variations removed.")

    ti_ctrl1, ti_ctrl2, _ = st.columns([2, 3, 2])
    with ti_ctrl1:
        ti_chart = st.selectbox("Chart type", CHART_TYPES, key="ti_chart")
    with ti_ctrl2:
        ti_n = safe_slider("Top N", 5, len(titles_df), len(titles_df), "ti_n")

    plot_result(titles_df, "Label", "Count",
                f"Top {ti_n} Standardized Job Titles", ti_chart, ti_n)
    st.markdown(
        '<p class="chart-note">'
        'AI Engineer and Backend Developer top the standardized title list. '
        'Senior-level presence confirms that experience is a key hiring criterion.'
        '</p>',
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════════════════
#  TAB 4 — LOCATION INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Location Insights")
    st.markdown('<span class="badge badge-pub">Published Results — Full Dataset</span>',
                unsafe_allow_html=True)
    st.caption("Geographic concentration of tech and data job postings across Saudi Arabia.")

    ci_ctrl1, ci_ctrl2, _ = st.columns([2, 3, 2])
    with ci_ctrl1:
        ci_chart = st.selectbox("Chart type", CHART_TYPES, key="ci_chart")
    with ci_ctrl2:
        ci_n = safe_slider("Top N", 3, len(cities_df), len(cities_df), "ci_n")

    plot_result(cities_df, "Label", "Count",
                f"Top {ci_n} Cities by Job Posting Volume", ci_chart, ci_n)
    st.markdown(
        '<p class="chart-note">'
        'Riyadh dominates with more than half of all postings. '
        'Jeddah and Dammam serve as secondary hubs in the Western and Eastern Province corridors.'
        '</p>',
        unsafe_allow_html=True,
    )

    st.divider()

    if "job_employment_type" in jobs_df.columns:
        st.markdown('<span class="sec-label">Employment Type Breakdown</span>', unsafe_allow_html=True)
        emp_counts = (
            jobs_df["job_employment_type"]
            .value_counts()
            .reset_index()
        )
        emp_counts.columns = ["Label", "Count"]

        if not emp_counts.empty:
            e1, e2 = st.columns(2)
            with e1:
                plot_result(emp_counts, "Label", "Count",
                            "Employment Type — Share", "Pie", len(emp_counts), height=340)
            with e2:
                plot_result(emp_counts, "Label", "Count",
                            "Employment Type — Volume", "Bar", len(emp_counts), height=340)
        else:
            st.warning("No employment type data available.")

# ════════════════════════════════════════════════════════════════════════════
#  TAB 5 — INTERACTIVE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Interactive Analysis")
    st.markdown('<span class="badge badge-live">Dynamic — Filtered Subset</span>',
                unsafe_allow_html=True)
    st.caption(
        "Charts are recomputed from raw postings based on the filters below. "
        "Skills use the same 33-term taxonomy as the published analysis. "
        "Role groups are inferred from raw titles via keyword matching."
    )

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.container(border=True):
        ia_f1, ia_f2, ia_f3 = st.columns(3)
        with ia_f1:
            ia_cities = st.multiselect(
                "City", sorted(jobs_df["job_city"].dropna().unique()),
                key="ia_city", placeholder="All cities",
            )
        with ia_f2:
            ia_emp = st.multiselect(
                "Employment Type",
                sorted(jobs_df["job_employment_type"].dropna().unique())
                if "job_employment_type" in jobs_df.columns else [],
                key="ia_emp", placeholder="All types",
            )
        with ia_f3:
            ia_titles_filter = st.multiselect(
                "Job Title",
                sorted(jobs_df["job_title"].dropna().unique())
                if "job_title" in jobs_df.columns else [],
                key="ia_title", placeholder="All titles",
            )

    # Apply filters
    ia_df = jobs_df.copy()
    if ia_cities:
        ia_df = ia_df[ia_df["job_city"].isin(ia_cities)]
    if ia_emp and "job_employment_type" in ia_df.columns:
        ia_df = ia_df[ia_df["job_employment_type"].isin(ia_emp)]
    if ia_titles_filter and "job_title" in ia_df.columns:
        ia_df = ia_df[ia_df["job_title"].isin(ia_titles_filter)]

    # ── Filtered KPI row ──────────────────────────────────────────────────────
    ia_k1, ia_k2, ia_k3, ia_k4 = st.columns(4)
    for col, val, label in [
        (ia_k1, f"{len(ia_df):,}",
                 "Postings in Scope"),
        (ia_k2, f"{ia_df['employer_name'].nunique():,}" if "employer_name" in ia_df.columns else "—",
                 "Companies"),
        (ia_k3, f"{ia_df['job_city'].nunique():,}"      if "job_city"      in ia_df.columns else "—",
                 "Cities"),
        (ia_k4, f"{ia_df['job_title'].nunique():,}"     if "job_title"     in ia_df.columns else "—",
                 "Unique Titles"),
    ]:
        col.markdown(
            f'<div class="kpi-block">'
            f'<div class="kpi-number">{val}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if ia_df.empty:
        st.warning("No records match the selected filters.")
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()

        # Column badge headers
        pub_col, live_col = st.columns(2)
        with pub_col:
            st.markdown(
                '<span class="badge badge-pub">Published — Full Dataset</span>',
                unsafe_allow_html=True,
            )
        with live_col:
            st.markdown(
                f'<span class="badge badge-live">Filtered — {len(ia_df):,} postings</span>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Skills ─────────────────────────────────────────────────────────────
        st.markdown('<span class="sec-label">Top Skills</span>', unsafe_allow_html=True)

        sk_c1, sk_c2, _ = st.columns([2, 3, 2])
        with sk_c1:
            ia_sk_chart = st.selectbox("Chart type", CHART_TYPES, key="ia_sk_chart")
        with sk_c2:
            live_skills_all = compute_live_skills(ia_df["clean_description"])
            ia_sk_n = safe_slider("Top N", 5, len(live_skills_all), 15, "ia_sk_n")

        sk_pub, sk_live = st.columns(2)
        with sk_pub:
            plot_result(skills_df, "Label", "Count",
                        "Published — Top Skills", ia_sk_chart, ia_sk_n, height=380)
        with sk_live:
            if live_skills_all.empty:
                st.warning("No skill matches found in the filtered postings.")
            else:
                plot_result(live_skills_all, "Label", "Count",
                            "Filtered — Top Skills", ia_sk_chart, ia_sk_n, height=380)

        st.caption(
            f"Re-extracted from job descriptions using the same {len(skills_df)}-term taxonomy. "
            "Only skills with at least one match are shown."
        )

        st.divider()

        # ── Cities ─────────────────────────────────────────────────────────────
        st.markdown('<span class="sec-label">Top Cities</span>', unsafe_allow_html=True)

        ia_ci_chart = st.selectbox("Chart type", CHART_TYPES, key="ia_ci_chart")

        live_cities_ia = (
            ia_df["job_city"].value_counts().reset_index()
        )
        live_cities_ia.columns = ["Label", "Count"]

        ci_pub, ci_live = st.columns(2)
        with ci_pub:
            plot_result(cities_df, "Label", "Count",
                        "Published — Top Cities", ia_ci_chart, 10, height=340)
        with ci_live:
            plot_result(live_cities_ia, "Label", "Count",
                        "Filtered — Cities", ia_ci_chart, 10, height=340)

        st.divider()

        # ── Job Titles ──────────────────────────────────────────────────────────
        st.markdown('<span class="sec-label">Top Job Titles</span>', unsafe_allow_html=True)

        ia_ti_chart = st.selectbox("Chart type", CHART_TYPES, key="ia_ti_chart")

        live_titles_ia = (
            ia_df["job_title"].value_counts().reset_index()
        )
        live_titles_ia.columns = ["Label", "Count"]

        ti_pub, ti_live = st.columns(2)
        with ti_pub:
            plot_result(titles_df, "Label", "Count",
                        "Published — Standardized Titles", ia_ti_chart, 10, height=360)
        with ti_live:
            plot_result(live_titles_ia, "Label", "Count",
                        "Filtered — Raw Title Frequency", ia_ti_chart, 10, height=360)

        st.caption(
            "Published titles are normalized to canonical forms. "
            "Filtered titles use raw values — expect more variation in naming."
        )

        st.divider()

        # ── Role Distribution ───────────────────────────────────────────────────
        st.markdown('<span class="sec-label">Role Distribution</span>', unsafe_allow_html=True)

        ia_ro_chart = st.selectbox("Chart type", CHART_TYPES, key="ia_ro_chart")

        live_roles_ia = compute_live_roles(ia_df["job_title"])

        ro_pub, ro_live = st.columns(2)
        with ro_pub:
            plot_result(roles_df, "Label", "Count",
                        "Published — Role Groups", ia_ro_chart, len(roles_df), height=360)
        with ro_live:
            if live_roles_ia.empty:
                st.warning("No roles classified from the filtered titles.")
            else:
                plot_result(live_roles_ia, "Label", "Count",
                            "Filtered — Role Groups", ia_ro_chart, len(live_roles_ia), height=360)

        st.caption(
            "Role groups inferred from raw titles via keyword matching. "
            "Unmatched titles are grouped under 'Other'."
        )

# ════════════════════════════════════════════════════════════════════════════
#  TAB 6 — DATA EXPLORER
# ════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### Data Explorer")
    st.caption(
        f"Full dataset — {len(jobs_df):,} job postings. "
        "Search by title or employer name to filter results."
    )

    search_query = st.text_input(
        "Search",
        placeholder="Search by job title or company name...",
        key="de_search",
        label_visibility="collapsed",
    )

    display_data = jobs_df.copy()
    if search_query.strip():
        q = search_query.strip()
        title_match = (
            display_data.get("job_title", pd.Series(dtype=str))
            .str.contains(q, case=False, na=False)
        )
        employer_match = (
            display_data.get("employer_name", pd.Series(dtype=str))
            .str.contains(q, case=False, na=False)
        )
        display_data = display_data[title_match | employer_match]

    display_cols = [c for c in [
        "job_title", "employer_name", "job_city",
        "job_employment_type", "job_is_remote", "job_posted_at",
    ] if c in display_data.columns]

    if display_data.empty:
        st.warning("No records match the search query.")
    else:
        st.dataframe(
            display_data[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=500,
        )

    with st.expander("Summary Statistics"):
        if display_data.empty:
            st.info("No data to summarise.")
        else:
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Records",
                      f"{len(display_data):,}")
            s2.metric("Unique Employers",
                      f"{display_data['employer_name'].nunique():,}"
                      if "employer_name" in display_data.columns else "—")
            s3.metric("Unique Cities",
                      f"{display_data['job_city'].nunique():,}"
                      if "job_city" in display_data.columns else "—")
            s4.metric("Unique Titles",
                      f"{display_data['job_title'].nunique():,}"
                      if "job_title" in display_data.columns else "—")

            if "job_employment_type" in display_data.columns:
                st.markdown('<span class="sec-label">Employment Type Breakdown</span>',
                            unsafe_allow_html=True)
                emp_summary = (
                    display_data["job_employment_type"]
                    .value_counts()
                    .reset_index()
                )
                emp_summary.columns = ["Type", "Postings"]
                st.dataframe(emp_summary, hide_index=True, use_container_width=True)

    st.caption(
        f"Dataset: {len(jobs_df):,} Saudi tech and data job postings, 2024. "
        "Columns shown are a display subset."
    )
