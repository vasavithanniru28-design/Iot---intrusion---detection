
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(
    page_title="IoT Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: #F3F4F6;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
[data-testid="stMetric"] {
    background-color: #1C293B;
    border: 1px solid #334155;
    padding: 18px;
    border-radius: 12px;
}
h1, h2, h3 {
    color: #F8FAFC;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = joblib.load("models/intrusion_model.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    feature_encoders = joblib.load("models/feature_encoders.pkl")
    return model, label_encoder, feature_encoders


def normalize_label(label):
    return str(label).strip().lower().rstrip(".")


def prepare_data(uploaded_file, model, feature_encoders):
    data = pd.read_csv(uploaded_file, header=None)

    if data.empty:
        raise ValueError("The uploaded CSV is empty.")

    if data.shape[1] != model.n_features_in_:
        raise ValueError(
            f"The model expects {model.n_features_in_} feature columns, "
            f"but the uploaded file contains {data.shape[1]} columns."
        )

    data.columns = range(data.shape[1])
    data = data.copy()

    for col, encoder in feature_encoders.items():
        col = int(col)

        if col not in data.columns:
            continue

        values = data[col].astype(str)
        known_values = set(encoder.classes_)
        unknown_values = set(values) - known_values

        if unknown_values:
            examples = list(unknown_values)[:5]
            raise ValueError(
                f"Unknown values in feature column {col}: {examples}. "
                "Please upload data compatible with the trained encoders."
            )

        data[col] = encoder.transform(values)

    data = data.apply(pd.to_numeric, errors="raise")

    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("The CSV contains missing or infinite values.")

    return data


def calculate_drift(reference, current):
    reference_mean = reference.mean()
    reference_std = reference.std().replace(0, 1).fillna(1)

    current_mean = current.mean()

    shifts = (
        (current_mean - reference_mean).abs()
        / reference_std
    )

    result = pd.DataFrame({
        "Feature": [f"Feature {c}" for c in shifts.index],
        "Standardized Mean Shift": shifts.values
    })

    average_shift = float(shifts.mean())

    return average_shift, result.sort_values(
        "Standardized Mean Shift",
        ascending=False
    )


st.title("🛡️ IoT Intrusion Detection System")
st.caption(
    "Lightweight Machine Learning | Confidence Analysis | "
    "Traffic Drift | Explainable Predictions"
)

st.info(
    "This academic prototype analyzes uploaded network traffic samples. "
    "It does not monitor a live network."
)

try:
    model, label_encoder, feature_encoders = load_model()
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()


st.sidebar.header("Detection Settings")

confidence_threshold = st.sidebar.slider(
    "Minimum confidence threshold",
    min_value=0.50,
    max_value=0.99,
    value=0.60,
    step=0.05,
    help="Predictions below this threshold are flagged for review."
)

st.sidebar.caption(
    "Low confidence means the model is less certain, "
    "not necessarily that its prediction is wrong."
)

st.divider()

st.subheader("📂 Upload Network Traffic Data")

uploaded_file = st.file_uploader(
    "Upload a CSV containing feature columns only, without the class label.",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Upload a compatible CSV file to start detection.")

    st.markdown("### Features in this prototype")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**1. Lightweight Detection**")
        st.write("Uses the existing trained Random Forest classifier.")

        st.markdown("**2. Confidence-Based Identification**")
        st.write("Flags predictions whose confidence is below your threshold.")

    with c2:
        st.markdown("**3. Traffic-Drift Monitoring**")
        st.write("Compares feature means across uploaded batches.")

        st.markdown("**4. Human-Readable Explanations**")
        st.write("Shows prediction confidence and important model features.")

    st.stop()


try:
    data = prepare_data(uploaded_file, model, feature_encoders)

    st.success("Dataset loaded and prepared successfully.")

    st.caption(
        f"File: {uploaded_file.name} | "
        f"Rows: {len(data)} | "
        f"Features per row: {data.shape[1]}"
    )

    # Measure prediction processing time.
    start_time = time.perf_counter()

    predictions = model.predict(data)
    probabilities = model.predict_proba(data)

    elapsed_seconds = time.perf_counter() - start_time

    predicted_labels = label_encoder.inverse_transform(
        predictions.astype(int)
    )

    confidence_scores = probabilities.max(axis=1) * 100

    normalized_labels = [
        normalize_label(label) for label in predicted_labels
    ]

    is_normal = np.array([
        label == "normal" for label in normalized_labels
    ])

    is_uncertain = confidence_scores < (
        confidence_threshold * 100
    )

    result_data = pd.DataFrame({
        "Traffic ID": range(1, len(data) + 1),
        "Predicted Class": predicted_labels,
        "Confidence (%)": np.round(confidence_scores, 2),
        "Review Status": np.where(
            is_uncertain,
            "Review Required",
            "Confidence Above Threshold"
        )
    })

    st.divider()
    st.subheader("📊 Detection Statistics")

    total_rows = len(data)
    attack_count = int((~is_normal).sum())
    normal_count = int(is_normal.sum())
    uncertain_count = int(is_uncertain.sum())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows Analyzed", total_rows)
    col2.metric("Predicted Attacks", attack_count)
    col3.metric("Predicted Normal", normal_count)
    col4.metric("Uncertain Predictions", uncertain_count)

    st.caption(
        "Attack counts include every predicted class other than Normal. "
        "These are predictions, not confirmed incidents."
    )

    st.divider()
    st.subheader("📈 Traffic Classification Chart")

    chart_data = pd.DataFrame({
        "Traffic Type": ["Predicted Attacks", "Predicted Normal"],
        "Count": [attack_count, normal_count]
    })

    chart = alt.Chart(chart_data).mark_bar(
        cornerRadiusEnd=6
    ).encode(
        x=alt.X(
            "Traffic Type:N",
            title="Traffic Type",
            axis=alt.Axis(labelAngle=0)
        ),
        y=alt.Y(
            "Count:Q",
            title="Number of Rows",
            scale=alt.Scale(
                domain=[0, max(total_rows, 1)]
            )
        ),
        color=alt.Color(
            "Traffic Type:N",
            scale=alt.Scale(
                domain=["Predicted Attacks", "Predicted Normal"],
                range=["#EF4444", "#22C55E"]
            ),
            legend=None
        ),
        tooltip=["Traffic Type", "Count"]
    ).properties(height=300)

    st.altair_chart(chart, use_container_width=True)

    st.divider()
    st.subheader("🎯 Confidence-Based Identification")

    conf_col1, conf_col2 = st.columns(2)

    with conf_col1:
        st.metric(
            "Average Prediction Confidence",
            f"{confidence_scores.mean():.2f}%"
        )

    with conf_col2:
        st.metric(
            "Below Confidence Threshold",
            uncertain_count
        )

    if uncertain_count:
        st.warning(
            f"{uncertain_count} prediction(s) are below the selected "
            "confidence threshold. Review them before drawing conclusions."
        )
    else:
        st.success(
            "All predictions meet the selected confidence threshold. "
            "This does not guarantee correctness."
        )

    st.divider()
    st.subheader("📋 Prediction Results")

    filter_option = st.selectbox(
        "Filter predictions",
        [
            "All Predictions",
            "Predicted Attacks",
            "Predicted Normal",
            "Review Required"
        ]
    )

    if filter_option == "Predicted Attacks":
        displayed_results = result_data.loc[~is_normal]
    elif filter_option == "Predicted Normal":
        displayed_results = result_data.loc[is_normal]
    elif filter_option == "Review Required":
        displayed_results = result_data.loc[is_uncertain]
    else:
        displayed_results = result_data

    st.dataframe(
        displayed_results,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Prediction Results",
        data=result_data.to_csv(index=False).encode("utf-8"),
        file_name="intrusion_detection_results.csv",
        mime="text/csv"
    )

    st.divider()
    st.subheader("🧠 Selective Human-Readable Explanation")

    selected_row = st.selectbox(
        "Choose a traffic record to explain",
        options=list(range(len(data))),
        format_func=lambda i: (
            f"Traffic {i + 1} — {predicted_labels[i]} "
            f"({confidence_scores[i]:.2f}% confidence)"
        )
    )

    generate_explanation = st.checkbox(
        "Generate explanation for this prediction"
    )

    if generate_explanation:
        explanation_start = time.perf_counter()

        selected_label = str(predicted_labels[selected_row])
        selected_confidence = float(confidence_scores[selected_row])
        selected_is_normal = is_normal[selected_row]
        selected_is_uncertain = is_uncertain[selected_row]

        st.markdown(f"**Predicted class:** {selected_label}")
        st.markdown(
            f"**Model confidence:** {selected_confidence:.2f}%"
        )

        if selected_is_uncertain:
            st.warning(
                "This prediction is below the configured confidence "
                "threshold and should be reviewed."
            )
        elif selected_is_normal:
            st.success(
                "The model classified this sample as Normal. "
                "This does not prove the traffic is safe."
            )
        else:
            st.error(
                "The model classified this sample as an attack. "
                "This is a prediction and requires appropriate validation."
            )

        importances = getattr(model, "feature_importances_", None)

        if importances is not None and len(importances) == data.shape[1]:
            top_indices = np.argsort(importances)[::-1][:5]

            explanation_table = pd.DataFrame({
                "Feature": [
                    f"Feature {i}" for i in top_indices
                ],
                "Selected Row Value": [
                    data.iloc[selected_row, i] for i in top_indices
                ],
                "Global Model Importance (%)": [
                    round(float(importances[i]) * 100, 2)
                    for i in top_indices
                ]
            })

            st.markdown("**Important features in the trained model**")
            st.dataframe(
                explanation_table,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Global feature importance describes the model overall. "
                "It does not establish which features caused this "
                "individual prediction. Feature names are column indices "
                "because the uploaded dataset has no feature headers."
            )
        else:
            st.info(
                "This model does not expose feature importance in a "
                "supported format. A separate explanation method "
                "would be needed."
            )

        explanation_elapsed = (
            time.perf_counter() - explanation_start
        )

        st.caption(
            f"Explanation processing time: "
            f"{explanation_elapsed * 1000:.3f} ms. "
            "This is a local timing measurement, not a benchmark."
        )

    st.divider()
    st.subheader("📉 Traffic-Drift Monitoring")

    st.write(
        "This prototype compares the current upload with the first "
        "uploaded batch using standardized changes in feature means."
    )

    current_batch_key = (
        f"{uploaded_file.name}:{uploaded_file.size}"
    )

    if "drift_reference" not in st.session_state:
        st.session_state.drift_reference = data.copy()
        st.session_state.drift_reference_name = uploaded_file.name
        st.session_state.drift_reference_key = current_batch_key

        st.info(
            "Reference batch established. Upload a different dataset "
            "to compare its feature patterns with this reference."
        )

    elif current_batch_key == st.session_state.drift_reference_key:
        st.info(
            "This is the reference batch. Upload a different CSV "
            "to check for possible traffic-pattern changes."
        )

    else:
        reference = st.session_state.drift_reference

        if list(reference.columns) != list(data.columns):
            st.warning(
                "The feature columns do not match the reference batch. "
                "Drift comparison is unavailable."
            )
        else:
            average_shift, drift_details = calculate_drift(
                reference, data
            )

            drift_col1, drift_col2 = st.columns(2)

            drift_col1.metric(
                "Average Standardized Mean Shift",
                f"{average_shift:.3f}"
            )

            drift_col2.metric(
                "Reference Batch",
                st.session_state.drift_reference_name
            )

            st.dataframe(
                drift_details.head(10),
                use_container_width=True,
                hide_index=True
            )

            if average_shift > 0.5:
                st.warning(
                    "Possible feature-distribution change detected "
                    "by the prototype's heuristic. Further statistical "
                    "testing is needed to confirm drift."
                )
            else:
                st.success(
                    "The heuristic did not flag a large average "
                    "feature-mean shift. This does not rule out drift."
                )

            st.caption(
                "The 0.5 threshold is an exploratory setting, not a "
                "validated drift threshold. Encoded categorical values "
                "and small batches can make this comparison unreliable. "
                "This check does not prove changes in attack behavior."
            )

    st.divider()
    st.subheader("⚡ Performance Measurements")

    perf1, perf2, perf3 = st.columns(3)

    perf1.metric(
        "Prediction Processing Time",
        f"{elapsed_seconds * 1000:.3f} ms"
    )

    perf2.metric(
        "Average Time per Row",
        f"{elapsed_seconds * 1000 / total_rows:.3f} ms"
    )

    perf3.metric(
        "Rows per Second",
        f"{total_rows / elapsed_seconds:.2f}"
        if elapsed_seconds > 0
        else "N/A"
    )

    st.caption(
        "Timing covers model prediction and probability calculation "
        "in this local run. It excludes file upload, preprocessing, "
        "and model loading. Results depend on the computer and batch size."
    )

    st.divider()
    st.subheader("ℹ️ Dataset and Prototype Limitations")

    st.write(
        "This project uses the NSL-KDD-derived dataset for an initial "
        "network-intrusion prototype. It is not a modern IoT-specific "
        "dataset, and the current upload workflow does not perform "
        "live network monitoring."
    )

    st.write(
        "Prediction confidence is not the same as accuracy. To calculate "
        "accuracy, precision, recall, F1-score, and false-positive rate, "
        "evaluate predictions against known ground-truth labels."
    )

except Exception as e:
    st.error(f"Could not process the uploaded CSV: {e}")

st.divider()
st.caption(
    "IoT Intrusion Detection System | Academic Research Prototype"
)
