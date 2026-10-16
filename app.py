"""Streamlit Job Market Prediction."""
import json
import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from predict import predict_df  # noqa: E402

st.set_page_config(
    page_title=" Linkedin Job Market Prediction",
    layout="wide",
)

def load_json(filename):
    path = os.path.join(ROOT, "models", filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def show_chart(data, title, horizontal=False):
    st.subheader(title)
    if not data or not data.get("labels") or not data.get("values"):
        st.info("No chart data available.")
        return
    chart_df = pd.DataFrame({"Category": data["labels"], "Value": data["values"]})
    chart_df = chart_df.set_index("Category")
    st.bar_chart(chart_df, horizontal=horizontal)

st.title("Linkedin Job Market Prediction")
st.caption("LinkedIn India job-posting analysis and applicant-demand prediction")

page = st.sidebar.radio(
    "Choose page",
    ["Dashboard", "Predict a Job", "Predict from CSV", "Model Report"],
)

if page == "Dashboard":
    try:
        dashboard = load_json("dashboard_data.json")
        kpi = dashboard.get("kpi", {})
        cols = st.columns(4)
        cols[0].metric("Total jobs", f'{kpi.get("total_jobs", 0):,}')
        cols[1].metric("Jobs with applicant count", f'{kpi.get("with_applicant_count", 0):,}')
        cols[2].metric("Average applicants", kpi.get("avg_applicants", "N/A"))
        cols[3].metric("Companies", f'{kpi.get("companies", 0):,}')

        cols = st.columns(3)
        cols[0].metric("Median applicants", kpi.get("median_applicants", "N/A"))
        cols[1].metric("Remote share", f'{kpi.get("remote_share", 0)}%')
        cols[2].metric("Jobs capped at 200 applicants", f'{kpi.get("pct_capped_200", 0)}%')

        left, right = st.columns(2)
        with left:
            show_chart(dashboard.get("top_roles"), "Top job role families")
            show_chart(dashboard.get("work_type"), "Work type")
            show_chart(dashboard.get("avg_by_role"), "Average applicants by role")
            show_chart(dashboard.get("avg_by_seniority"), "Average applicants by seniority")
        with right:
            show_chart(dashboard.get("top_cities"), "Top cities")
            show_chart(dashboard.get("seniority"), "Seniority levels")
            show_chart(dashboard.get("avg_by_work"), "Average applicants by work type")
            show_chart(dashboard.get("avg_by_city"), "Average applicants by city")
        show_chart(dashboard.get("top_industries"), "Top industries")
        show_chart(dashboard.get("top_companies"), "Top companies")
        show_chart(dashboard.get("avg_by_age"), "Average applicants by posting age")
    except FileNotFoundError:
        st.error("Missing models/dashboard_data.json. Check your project folder.")

elif page == "Predict a Job":
    st.subheader("Predict applicant demand for one job")
    with st.form("job_prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            job_title = st.text_input("Job title", placeholder="e.g. Python Developer")
            company_name = st.text_input("Company name")
            location = st.text_input("Location", placeholder="e.g. Chennai, India")
            work_type = st.selectbox("Work type", ["", "On-site", "Remote", "Hybrid"])
            employment_type = st.selectbox(
                "Employment type",
                ["", "Full-time", "Part-time", "Contract", "Internship", "Temporary"],
            )
            seniority = st.selectbox(
                "Seniority",
                ["", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive"],
            )
        with c2:
            company_size = st.text_input("Company size", placeholder="e.g. 1,001-5,000")
            industry = st.text_input("Industry", placeholder="e.g. Software Development")
            posted_hours_ago = st.number_input("Hours since posted", min_value=0.0, value=24.0)
            alumni_count = st.number_input("Company alumni count", min_value=0, value=0, step=1)
            company_followers = st.number_input("Company followers", min_value=0, value=0, step=1)
            hiring_person_listed = st.selectbox("Hiring person listed?", [0, 1])
        description = st.text_area("Job description", height=140)
        submitted = st.form_submit_button("Predict applicants", type="primary")

    if submitted:
        if not job_title.strip():
            st.warning("Please enter a job title.")
        else:
            row = pd.DataFrame([{
                "job_title": job_title,
                "company_name": company_name or None,
                "location": location or None,
                "work_type": work_type or None,
                "employment_type": employment_type or None,
                "seniority": seniority or None,
                "company_size": company_size or None,
                "industry": industry or None,
                "posted_hours_ago": posted_hours_ago,
                "alumni_count": alumni_count if alumni_count else None,
                "company_followers": company_followers if company_followers else None,
                "hiring_person_listed": hiring_person_listed,
                "description": description or None,
            }])
            try:
                result = predict_df(row).iloc[0]
                a, b, c = st.columns(3)
                a.metric("Predicted applicants", int(result["predicted_applicants"]))
                b.metric("Estimated lower range", int(result["range_low_80pct"]))
                c.metric("Estimated upper range", int(result["range_high_80pct"]))
                st.success(f'Demand level: {result["demand_level"]}')
                st.write("Detected role:", result["role_family_detected"])
                st.write("Detected city:", result["city_detected"])
                st.caption("This is an estimate, not a guarantee. Applicant counts can vary.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.info("Confirm that models/best_model.joblib exists and was trained for this project.")

elif page == "Predict from CSV":
    st.subheader("Batch prediction from a CSV file")
    st.write("Upload a CSV with job details. The original dataset format and the template format are supported.")
    sample_path = os.path.join(ROOT, "sample", "sample_input.csv")
    if os.path.isfile(sample_path):
        with open(sample_path, "rb") as sample_file:
            st.download_button(
                "Download sample CSV template",
                data=sample_file.read(),
                file_name="sample_input.csv",
                mime="text/csv",
            )
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None:
        try:
            input_df = pd.read_csv(uploaded)
            st.write(f"Uploaded rows: {len(input_df):,}")
            with st.expander("Preview uploaded data"):
                st.dataframe(input_df.head(20), use_container_width=True)
            if st.button("Run CSV prediction", type="primary"):
                result_df = predict_df(input_df)
                st.success(f"Predictions completed for {len(result_df):,} rows.")
                summary_cols = [
                    c for c in [
                        "job_title", "job", "role_family_detected", "city_detected",
                        "predicted_applicants", "range_low_80pct",
                        "range_high_80pct", "demand_level",
                    ] if c in result_df.columns
                ]
                st.dataframe(result_df[summary_cols].head(200), use_container_width=True)
                counts = result_df["demand_level"].value_counts().rename_axis("Demand level").to_frame("Jobs")
                st.subheader("Demand level summary")
                st.bar_chart(counts)
                st.download_button(
                    "Download all predictions",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="job_predictions.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")

elif page == "Model Report":
    st.subheader("Model performance report")
    try:
        report = load_json("model_report.json")
        st.metric("Best model", report.get("best_model", "Not available"))
        st.write("Training rows:", report.get("n_train_rows", "Not available"))
        results = report.get("results", {})
        rows = []
        for model_name, values in results.items():
            test = values.get("test", {})
            rows.append({
                "Model": model_name,
                "Test R² (log)": test.get("r2_log"),
                "Test MAE (applicants)": test.get("mae_applicants"),
                "Test RMSE (applicants)": test.get("rmse_applicants"),
                "Demand level accuracy": test.get("level_accuracy"),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        importance = report.get("importance", [])
        if importance:
            imp_df = pd.DataFrame(importance)
            st.subheader("Feature importance")
            st.bar_chart(imp_df.set_index("feature")[["importance"]])
        scatter = report.get("scatter", [])
        if scatter:
            st.subheader("Actual vs predicted applicants (sample)")
            st.scatter_chart(pd.DataFrame(scatter), x="actual", y="pred")
    except FileNotFoundError:
        st.error("Missing models/model_report.json. Check your project folder.")
