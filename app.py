
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import os
import uuid

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


def read_dataset(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Only CSV and XLSX files are allowed")


def chart_json(fig):
    return pio.to_json(fig)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not (file.filename.endswith(".csv") or file.filename.endswith(".xlsx")):
            return JSONResponse({"error": "Only CSV and XLSX files are allowed"}, status_code=400)

        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, file_id + "_" + file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        df = read_dataset(file_path)

        preview = df.head(20).replace({np.nan: None}).to_dict(orient="records")

        return {
            "file_id": file_path,
            "rows": df.shape[0],
            "columns_count": df.shape[1],
            "columns": list(df.columns),
            "preview": preview
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/analyze")
async def analyze_dataset(data: dict):
    try:
        file_path = data.get("file_id")
        df = read_dataset(file_path)

        original_shape = df.shape

        # Cleaning
        df = df.drop_duplicates()

        # Safe numeric conversion
        for col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > len(df) * 0.7:
                df[col] = converted

        # Fill missing values
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("Unknown")

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        charts = []

        # Histogram
        if len(numeric_cols) >= 1:
            col = numeric_cols[0]
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            charts.append({
                "title": f"Histogram: {col}",
                "json": chart_json(fig)
            })

        # Box plot
        if len(numeric_cols) >= 1:
            col = numeric_cols[0]
            fig = px.box(df, y=col, title=f"Box Plot of {col}")
            charts.append({
                "title": f"Box Plot: {col}",
                "json": chart_json(fig)
            })

        # Bar chart
        if len(categorical_cols) >= 1:
            col = categorical_cols[0]
            counts = df[col].value_counts().head(10).reset_index()
            counts.columns = [col, "count"]
            fig = px.bar(counts, x=col, y="count", title=f"Top Categories in {col}")
            charts.append({
                "title": f"Bar Chart: {col}",
                "json": chart_json(fig)
            })

        # Pie chart
        if len(categorical_cols) >= 1:
            col = categorical_cols[0]
            counts = df[col].value_counts().head(7).reset_index()
            counts.columns = [col, "count"]
            fig = px.pie(counts, names=col, values="count", title=f"Category Share: {col}")
            charts.append({
                "title": f"Pie Chart: {col}",
                "json": chart_json(fig)
            })

        # Scatter plot WITHOUT regression line
        if len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
            charts.append({
                "title": f"Scatter Plot: {x_col} vs {y_col}",
                "json": chart_json(fig)
            })

        # Correlation heatmap - safe
        top_correlations = []
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr(numeric_only=True)

            try:
                fig = px.imshow(corr.astype(float), title="Correlation Heatmap")
                charts.append({
                    "title": "Correlation Heatmap",
                    "json": chart_json(fig)
                })
            except Exception:
                pass

            corr_pairs = corr.abs().unstack().sort_values(ascending=False)
            corr_pairs = corr_pairs[corr_pairs < 1]

            seen = set()
            for (a, b), value in corr_pairs.items():
                pair = tuple(sorted([a, b]))
                if pair not in seen:
                    seen.add(pair)
                    top_correlations.append({
                        "column_1": a,
                        "column_2": b,
                        "correlation": round(float(corr.loc[a, b]), 3)
                    })
                if len(top_correlations) == 5:
                    break

        stats = {}
        if len(numeric_cols) >= 1:
            stats = df[numeric_cols].describe().round(2).to_dict()

        data_types = {col: str(df[col].dtype) for col in df.columns}
        missing_values = df.isnull().sum().to_dict()

        insights = []
        insights.append(f"The original dataset had {original_shape[0]} rows and {original_shape[1]} columns.")
        insights.append(f"After cleaning, the dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
        insights.append(f"The system detected {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns.")

        if len(numeric_cols) >= 1:
            for col in numeric_cols[:3]:
                insights.append(
                    f"The average value of '{col}' is {round(df[col].mean(), 2)}, "
                    f"with minimum {round(df[col].min(), 2)} and maximum {round(df[col].max(), 2)}."
                )

        if len(categorical_cols) >= 1:
            col = categorical_cols[0]
            top_category = df[col].value_counts().idxmax()
            top_count = int(df[col].value_counts().max())
            insights.append(f"The most common category in '{col}' is '{top_category}', appearing {top_count} times.")

        if len(top_correlations) >= 1:
            c = top_correlations[0]
            insights.append(
                f"The strongest correlation is between '{c['column_1']}' and '{c['column_2']}' "
                f"with correlation value {c['correlation']}."
            )

        insights.append("Recommendation: use the strongest patterns, frequent categories, and unusual values for data-driven decision making.")

        conclusion = "The application successfully uploads, cleans, analyzes, visualizes, and summarizes the dataset using an automated data visualization pipeline."

        return {
            "summary": {
                "original_shape": original_shape,
                "cleaned_shape": df.shape,
                "missing_values": missing_values,
                "data_types": data_types,
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "datetime_columns": []
            },
            "statistics": stats,
            "top_correlations": top_correlations,
            "charts": charts,
            "insights": insights,
            "conclusion": conclusion
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
