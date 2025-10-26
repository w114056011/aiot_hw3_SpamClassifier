"""
Minimal Streamlit app for model inspection and live inference.

Usage:
    streamlit run tools/streamlit_app.py

Features:
- Load a saved model pipeline (joblib). If a pipeline with vectorizer is saved, it will be used directly.
- Show evaluation metrics from model folder's report.json if available.
- Single-text inference (shows predicted label and probability when available).
- CSV batch inference (upload a CSV with `text` or `text_clean` column) and download predictions.

This is intentionally small and dependency-light.
"""

import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


def load_report(model_dir: Path):
    rpt = model_dir / "report.json"
    if rpt.exists():
        with open(rpt, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_model(path: Path):
    if not path.exists():
        st.warning(f"Model not found: {path}")
        return None
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def predict_single(model, text: str):
    try:
        pred = model.predict([text])[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([text])[0][1]
        return pred, proba
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None


def compute_scores(model, texts):
    """Return (pred_labels, scores) where scores is probability or decision score if available."""
    X = list(map(str, texts))
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)
        return preds, proba
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # scale scores to 0-1 for display using min-max
        min_s, max_s = scores.min(), scores.max()
        if max_s > min_s:
            proba = (scores - min_s) / (max_s - min_s)
        else:
            proba = (scores - min_s)
        preds = (proba >= 0.5).astype(int)
        return preds, proba
    else:
        preds = model.predict(X)
        return preds, None


def label_name(model, val):
    """Return a human-friendly label name for a predicted value.

    Heuristics:
    - If prediction is a string and model.classes_ are strings, return it.
    - If predictions are numeric (0/1), map 1->'spam', 0->'ham'.
    - If model.classes_ contains string labels, try to map indices.
    """
    try:
        # prefer obvious string predictions
        if isinstance(val, str):
            return val

        # handle numpy ints
        if hasattr(val, "item"):
            try:
                v = int(val.item())
            except Exception:
                v = val
        else:
            v = val

        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            # if classes are string labels, and value is an index, map
            if all(isinstance(c, str) for c in classes):
                try:
                    # if prediction is numeric index
                    return classes[int(v)]
                except Exception:
                    # if prediction equals one of the class strings
                    if str(v) in classes:
                        return str(v)
            # if classes are numeric 0/1, map to spam/ham
            if all(isinstance(c, (int, np.integer)) for c in classes):
                return "spam" if int(v) == 1 else "ham"

        # fallback: numeric -> spam/ham, else stringified
        if isinstance(v, (int, np.integer)):
            return "spam" if int(v) == 1 else "ham"
        return str(v)
    except Exception:
        return str(val)


def read_csv_flexible(source):
    """Read a CSV from a path or file-like and handle headerless label,text formats.

    Returns a DataFrame. Heuristics:
    - Try normal pd.read_csv()
    - If no 'text'/'text_clean' column, try reading with header=None and assign ['label','text',...]
    - If source is file-like, seek(0) before re-reading
    """
    is_path = isinstance(source, (str, Path))
    df = None
    try:
        df = pd.read_csv(source)
    except Exception:
        df = None

    def has_text_col(d):
        if d is None:
            return False
        cols = [str(c).lower() for c in d.columns]
        return any(c in ("text", "text_clean") for c in cols)

    if has_text_col(df):
        return df

    # Try reading as headerless data (label,text,...)
    try:
        if not is_path and hasattr(source, "seek"):
            try:
                source.seek(0)
            except Exception:
                pass
        df2 = pd.read_csv(source, header=None)
        if df2 is not None and df2.shape[1] >= 2:
            names = ["label", "text"] + [f"c{i}" for i in range(2, df2.shape[1])]
            df2.columns = names[: df2.shape[1]]
            return df2
    except Exception:
        pass

    return df


def sample_random(label_target: str):
    """Callback to sample a random example of label_target ('ham' or 'spam') from last_df.

    This function is safe to call as an on_click callback for Streamlit buttons.
    It reads `last_df`, `last_df_text_col`, `last_df_label_col` from session_state and writes
    `input_text` into session_state.
    """
    try:
        df_last = st.session_state.get('last_df')
        text_col_last = st.session_state.get('last_df_text_col')
        label_col_last = st.session_state.get('last_df_label_col')
        if df_last is None or text_col_last is None:
            st.warning("No dataset loaded to sample from. Load a CSV in the sidebar first.")
            return

        # normalize label values
        try:
            if label_col_last and label_col_last in df_last.columns:
                labels_norm = df_last[label_col_last].map(lambda s: 'spam' if str(s).strip().lower() in ('spam','1','true') else 'ham')
            else:
                labels_norm = None
        except Exception:
            labels_norm = None

        if labels_norm is None:
            st.warning("Label column not available to sample. Please select a label column in the sidebar.")
            return

        target_rows = df_last[labels_norm == label_target]
        if len(target_rows) == 0:
            st.warning(f"No {label_target} examples found in the loaded dataset.")
            return

        sample_text = target_rows.sample(1)[text_col_last].iloc[0]
        st.session_state['input_text'] = str(sample_text)
        # best-effort rerun if available
        if hasattr(st, "experimental_rerun"):
            try:
                st.experimental_rerun()
            except Exception:
                pass
    except Exception as e:
        st.error(f"Sampling failed: {e}")


def on_csv_select_change():
    """Callback when the CSV selectbox changes; mark a flag so UI can react."""
    try:
        # mark that the selection changed
        st.session_state['csv_select_changed'] = True
        # clear any previously-loaded dataframe so we force a fresh read
        if 'last_df' in st.session_state:
            try:
                st.session_state.pop('last_df')
            except Exception:
                st.session_state['last_df'] = None
        # clear stored loaded path so downstream logic uses the new selection
        st.session_state['loaded_csv_path'] = None
        # best-effort: trigger a rerun so the UI immediately recomputes predictions/plots
        if hasattr(st, "experimental_rerun"):
            try:
                st.experimental_rerun()
            except Exception:
                # if rerun fails, we still rely on Streamlit's normal rerun behavior
                pass
    except Exception:
        pass


def plot_roc_pr(y_true, scores):
    figs = {}
    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        fig1, ax1 = plt.subplots()
        ax1.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
        ax1.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC Curve")
        ax1.legend()
        figs["roc"] = fig1
    except Exception:
        figs["roc"] = None

    try:
        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(recall, precision)
        fig2, ax2 = plt.subplots()
        ax2.plot(recall, precision, label=f"PR (AUC = {pr_auc:.3f})")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve")
        ax2.legend()
        figs["pr"] = fig2
    except Exception:
        figs["pr"] = None

    return figs


def run():
    st.set_page_config(page_title="Spam Classifier")
    st.title("Spam/Ham Classifier")

    default_model = Path("models/logreg/model.joblib")
    model_path = st.sidebar.text_input("Model path", str(default_model))
    model_dir = Path(model_path).parent
    # Auto-load model at startup if present
    if "model" not in st.session_state:
        st.session_state['model'] = None
        st.session_state['report'] = None
        p = Path(model_path)
        if p.exists():
            st.session_state['model'] = load_model(p)
            st.session_state['report'] = load_report(p.parent)

    if st.sidebar.button("Reload model"):
        st.session_state['model'] = load_model(Path(model_path))
        st.session_state['report'] = load_report(model_dir)

    if st.session_state['report']:
        st.sidebar.subheader("Metrics")
        # Show a small selection of metrics if present
        for k in ("precision", "recall", "f1", "roc_auc"):
            if k in st.session_state['report']:
                st.sidebar.write(f"{k}: {st.session_state['report'][k]}")

    st.header("Single text inference")

    # Random example buttons (use last loaded dataframe if available)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Random ham example"):
            df_last = st.session_state.get('last_df')
            text_col_last = st.session_state.get('last_df_text_col')
            label_col_last = st.session_state.get('last_df_label_col')
            if df_last is None or text_col_last is None:
                st.warning("No dataset loaded to sample from. Load a CSV in the sidebar first.")
            else:
                # normalize label values and sample
                try:
                    labels_norm = df_last[label_col_last].map(lambda s: 'spam' if str(s).strip().lower() in ('spam','1','true') else 'ham') if label_col_last in df_last.columns else None
                except Exception:
                    labels_norm = None
                if labels_norm is None:
                    st.warning("Label column not available to sample ham. Please select a label column in the sidebar.")
                else:
                    ham_rows = df_last[labels_norm == 'ham']
                    if len(ham_rows) == 0:
                        st.warning("No ham examples found in the loaded dataset.")
                    else:
                        sample_text = ham_rows.sample(1)[text_col_last].iloc[0]
                        st.session_state['input_text'] = str(sample_text)
                        # some streamlit versions expose experimental_rerun; others do not
                        if hasattr(st, "experimental_rerun"):
                            try:
                                st.experimental_rerun()
                            except Exception:
                                pass
    with col2:
        if st.button("Random spam example"):
            df_last = st.session_state.get('last_df')
            text_col_last = st.session_state.get('last_df_text_col')
            label_col_last = st.session_state.get('last_df_label_col')
            if df_last is None or text_col_last is None:
                st.warning("No dataset loaded to sample from. Load a CSV in the sidebar first.")
            else:
                try:
                    labels_norm = df_last[label_col_last].map(lambda s: 'spam' if str(s).strip().lower() in ('spam','1','true') else 'ham') if label_col_last in df_last.columns else None
                except Exception:
                    labels_norm = None
                if labels_norm is None:
                    st.warning("Label column not available to sample spam. Please select a label column in the sidebar.")
                else:
                    spam_rows = df_last[labels_norm == 'spam']
                    if len(spam_rows) == 0:
                        st.warning("No spam examples found in the loaded dataset.")
                    else:
                        sample_text = spam_rows.sample(1)[text_col_last].iloc[0]
                        st.session_state['input_text'] = str(sample_text)
                        if hasattr(st, "experimental_rerun"):
                            try:
                                st.experimental_rerun()
                            except Exception:
                                pass

    # persist the input text in session state and render the text area after sampling buttons
    if 'input_text' not in st.session_state:
        st.session_state['input_text'] = ''
    text = st.text_area("Enter text to classify", value=st.session_state.get('input_text', ''), height=120, key='input_text')

    if st.button("Predict"):
        if st.session_state['model'] is None:
            st.error("No model loaded. Click 'Load model' in the sidebar or ensure the default path is correct.")
        else:
            pred, proba = predict_single(st.session_state['model'], st.session_state['input_text'])
            # show human-friendly name for the predicted class
            st.write("Predicted:", label_name(st.session_state['model'], pred))
            if proba is not None:
                st.write("Predicted probability (spam):", float(proba))

    st.header("Batch inference (CSV upload or demo data)")

    # Move CSV controls to the sidebar and allow selecting files from data/ by default
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"] )
    data_dir = Path("data")
    csv_choices = ["-- select --"]
    # Also include CSVs from data/processed/ in the same selector
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        csv_choices += [str(p) for p in sorted(processed_dir.glob("*.csv"))]
    if data_dir.exists():
        # list csv files in the data directory
        csv_choices += [str(p) for p in sorted(data_dir.glob("*.csv"))]
    # default to the first data file if present
    default_index = 1 if len(csv_choices) > 1 else 0
    selected_csv = st.sidebar.selectbox("Or choose CSV from data/ (includes data/processed)", csv_choices, index=default_index, key='selected_csv', on_change=on_csv_select_change)
    demo_load = st.sidebar.checkbox("Load demo dataset on start", value=True)

    df = None
    # priority: uploaded file > selected data file > selected processed > default demo file
    if uploaded is not None:
        try:
            df = read_csv_flexible(uploaded)
            if df is None:
                raise RuntimeError("could not parse uploaded CSV")
            st.sidebar.info("Using uploaded CSV")
        except Exception as e:
            st.sidebar.error(f"Failed to read uploaded CSV: {e}")
    elif selected_csv != "-- select --":
        try:
            df = read_csv_flexible(selected_csv)
            if df is None:
                raise RuntimeError(f"could not parse {selected_csv}")
            if demo_load:
                st.info(f"Loaded dataset from {selected_csv}")
        except Exception as e:
            st.error(f"Failed to read {selected_csv}: {e}")
    
    else:
        demo_path = Path("data/processed/sms_spam_clean.csv")
        if demo_path.exists() and demo_load:
            try:
                df = read_csv_flexible(demo_path)
                if df is None:
                    raise RuntimeError("could not parse demo dataset")
                st.info(f"Loaded demo dataset from {demo_path}")
            except Exception as e:
                st.error(f"Failed to load demo dataset: {e}")

    if df is not None:
        # Provide manual field selection in the sidebar using detected columns
        cols = list(df.columns)
        # determine sensible defaults
        default_text = None
        for candidate in ("text_clean", "text", "message", "body"):
            if candidate in cols:
                default_text = candidate
                break
        if default_text is None:
            # choose the first column that looks like text (string dtype or index 1)
            default_text = cols[1] if len(cols) > 1 else cols[0]

        text_col = st.sidebar.selectbox("Text column", cols, index=cols.index(default_text) if default_text in cols else 0)

        # label/ground-truth column selection
        label_candidates = [c for c in cols if c.lower() in ("label", "class", "target", "y", "truth")]
        default_label = label_candidates[0] if label_candidates else (cols[0] if cols else None)
        label_col = st.sidebar.selectbox("Label column (optional)", ["(none)"] + cols, index=0 if default_label is None else (cols.index(default_label) + 1))

        if text_col is None:
            st.error("CSV must contain a text column; please select it in the sidebar")
            return
        # normalize label_col value
        # normalize label_col value
        if label_col == "(none)":
            label_col = None

        # store the loaded dataframe and chosen columns for the random-sample buttons
        st.session_state['last_df'] = df
        st.session_state['last_df_text_col'] = text_col
        st.session_state['last_df_label_col'] = label_col
        # store which file was loaded so UI can react when switching datasets
        try:
            if uploaded is not None:
                st.session_state['loaded_csv_path'] = getattr(uploaded, 'name', 'uploaded')
            else:
                st.session_state['loaded_csv_path'] = str(selected_csv)
        except Exception:
            st.session_state['loaded_csv_path'] = str(selected_csv)

        try:
            if st.session_state['model'] is None:
                st.error("No model loaded; cannot run batch inference")
            else:
                X = df[text_col].astype(str).tolist()
                preds, scores = compute_scores(st.session_state['model'], X)
                if scores is not None:
                    df["pred_proba"] = scores
                # map predicted values to human-friendly names
                try:
                    df["pred_label"] = [label_name(st.session_state['model'], v) for v in preds]
                except Exception:
                    df["pred_label"] = preds

                # determine which label column to use for evaluation: prefer user-chosen, else try heuristics
                label_col_found = label_col if (label_col and label_col in df.columns) else None
                if label_col_found is None:
                    for c in ("label", "Label", "truth", "y", "target", "class"):
                        if c in df.columns:
                            label_col_found = c
                            break

                st.subheader("Predictions preview")
                st.dataframe(df.head(50))
                csv = df.to_csv(index=False)
                st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv")

                if label_col_found is not None and scores is not None:
                    y_true = df[label_col_found].map(lambda s: 1 if str(s).strip().lower() in ("spam", "1", "true") else 0).astype(int)
                    # show ROC/PR
                    figs = plot_roc_pr(y_true, scores)
                    if figs.get("roc") is not None:
                        st.subheader("ROC Curve")
                        st.pyplot(figs["roc"])
                    if figs.get("pr") is not None:
                        st.subheader("Precision-Recall Curve")
                        st.pyplot(figs["pr"])

                    # threshold slider
                    thresh = st.sidebar.slider("Probability threshold", 0.0, 1.0, 0.5, 0.01)
                    y_pred_thresh = (scores >= thresh).astype(int)
                    prec = precision_score(y_true, y_pred_thresh)
                    rec = recall_score(y_true, y_pred_thresh)
                    f1 = f1_score(y_true, y_pred_thresh)
                    st.metric("Precision", f"{prec:.3f}")
                    st.metric("Recall", f"{rec:.3f}")
                    st.metric("F1", f"{f1:.3f}")

                    # confusion matrix plot
                    try:
                        cm = confusion_matrix(y_true, y_pred_thresh)
                        fig_cm, ax_cm = plt.subplots()
                        ax_cm.matshow(cm, cmap="Blues")
                        for (i, j), z in np.ndenumerate(cm):
                            ax_cm.text(j, i, str(z), ha="center", va="center")
                        ax_cm.set_xlabel("Predicted")
                        ax_cm.set_ylabel("Actual")
                        ax_cm.set_xticks([0, 1])
                        ax_cm.set_yticks([0, 1])
                        ax_cm.set_xticklabels(["ham", "spam"])
                        ax_cm.set_yticklabels(["ham", "spam"])
                        st.subheader("Confusion Matrix")
                        st.pyplot(fig_cm)
                    except Exception:
                        pass
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")


if __name__ == "__main__":
    run()
