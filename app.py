import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crop Recommendation System", layout="centered")

st.title("AI-Based Crop Recommendation System")
st.write(
    "This dashboard shows dual-scored crop recommendations per field, "
    "balancing sustainability and expected yield."
)

@st.cache_data
def load_recommendations():
    return pd.read_csv("crop_recommendations.csv")

df = load_recommendations()

if df.empty:
    st.error("No recommendations found. Run rank_crops.py first.")
else:
    field_ids = df["field_id"].unique()
    selected_field = st.sidebar.selectbox("Select Field ID", field_ids)

    field_data = df[df["field_id"] == selected_field].sort_values(
        "final_score", ascending=False
    )

    st.header(f"Top Crops for Field {selected_field}")

    st.subheader("Ranked Recommendations")
    st.dataframe(field_data.reset_index(drop=True))

    # Final score bar chart
    st.subheader("Final Score per Crop")
    fig, ax = plt.subplots()
    ax.bar(field_data["crop"], field_data["final_score"])
    ax.set_xlabel("Crop")
    ax.set_ylabel("Final Score")
    ax.set_title(f"Final Scores – Field {selected_field}")
    st.pyplot(fig)

    # Sustainability vs Yield chart
    st.subheader("Sustainability vs Yield (Scores)")
    fig2, ax2 = plt.subplots()
    x = range(len(field_data))
    width = 0.35

    ax2.bar(x, field_data["sustainability_score"], width, label="Sustainability")
    ax2.bar(
        [i + width for i in x],
        field_data["yield_score"],
        width,
        label="Yield",
    )

    ax2.set_xticks([i + width / 2 for i in x])
    ax2.set_xticklabels(field_data["crop"], rotation=0)
    ax2.set_ylabel("Score")
    ax2.legend()
    st.pyplot(fig2)

    st.success("Dashboard loaded successfully")
