import streamlit as st
import numpy as np
import pandas as pd
import joblib
import re

# PAGE CONFIG
st.set_page_config(
    page_title="Drug Recommendation System",
    page_icon="🩺",
    layout="centered"
)

# MODERN UI
st.markdown("""
<style>

.block-container{
    max-width:900px;
    padding-top:2rem;
    padding-bottom:2rem;
}

/* Background */
.stApp{
    background-color:#0f172a;
}

/* Header */
.main-title{
    font-size:3rem;
    font-weight:800;
    color:white;
    margin-bottom:0.2rem;
    letter-spacing:-1px;
}

.subtitle{
    color:#94a3b8;
    font-size:1rem;
    margin-bottom:2rem;
}

/* Main Cards */
.card{
    background:#111827;
    border:1px solid rgba(255,255,255,0.08);
    border-radius:20px;
    padding:1.4rem;
    margin-top:1rem;
    margin-bottom:1rem;
    box-shadow:0 4px 20px rgba(0,0,0,0.25);
}

/* Drug Cards */
.drug-card{
    background:#172033;
    border:1px solid rgba(255,255,255,0.08);
    border-radius:16px;
    padding:0.9rem 1rem;
    margin-bottom:0.8rem;
    font-size:1rem;
    font-weight:600;
    color:white;
}

/* Review Box */
.review-box{
    line-height:1.8;
    font-size:0.96rem;
    color:#d1d5db;
}

/* Textarea */
textarea{
    border-radius:16px !important;
    background:#111827 !important;
    color:white !important;
    border:1px solid rgba(255,255,255,0.08) !important;
}

/* Button */
.stButton > button{
    height:3.2rem;
    border-radius:14px;
    font-size:1rem;
    font-weight:700;
    border:none;
    background:linear-gradient(
        90deg,
        #2563eb,
        #1d4ed8
    );
    color:white;
}

.stButton > button:hover{
    opacity:0.92;
}

/* Progress */
.stProgress > div > div > div > div{
    border-radius:20px;
}

/* Expander */
.streamlit-expanderHeader{
    font-size:1rem;
    font-weight:600;
}

/* Spacing */
.element-container:has(.stButton){
    margin-top:0.5rem;
}

</style>
""", unsafe_allow_html=True)

# LOAD RESOURCES
@st.cache_resource
def load_resources():

    model = joblib.load("../Task5/Model/tuned_model.pkl")
    tfidf = joblib.load("../Task2/Model/tfidf.pkl")
    le = joblib.load("../Task2/Model/label_encoder.pkl")
    data = pd.read_csv("../Task1/Clean Data/train_cleaned.csv")

    return model, tfidf, le, data


model, tfidf, le, data = load_resources()

CONFIDENCE_THRESHOLD = 0.60

# CLEAN TEXT
def clean(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# VALIDATION
def is_valid_input(text):

    text = clean(text)

    words = text.split()

    if len(words) < 2:
        return False

    vec = tfidf.transform([text])

    return vec.nnz >= 2


# PREDICTION
def predict(text):

    text = clean(text)

    text_vec = tfidf.transform([text])

    pred = model.predict(text_vec)[0]

    proba = model.predict_proba(text_vec)[0]

    confidence = float(np.max(proba))

    condition = le.inverse_transform(
        [pred]
    )[0]

    return condition, confidence, proba


# DRUG RECOMMENDATION
def get_recommendations(condition):

    subset = data[
        data["condition"] == condition
    ]

    grouped = subset.groupby(
        "drugName"
    ).agg({
        "rating": "mean",
        "usefulCount": "mean",
        "review": "count"
    })

    grouped.columns = [
        "rating",
        "useful",
        "reviews"
    ]

    # Keep drugs with enough reviews
    grouped = grouped[
        grouped["reviews"] >= 5
    ]

    # Ranking score
    grouped["score"] = (
        grouped["rating"] * 0.7 +
        (grouped["useful"] / 100) * 0.3
    )

    grouped = grouped.sort_values(
        "score",
        ascending=False
    )

    return grouped.head(5)


# GET REVIEWS FOR SELECTED DRUG
def get_drug_reviews(drug):

    subset = data[
        data["drugName"] == drug
    ].copy()

    subset = subset.dropna(
        subset=["review"]
    )

    subset["review_length"] = (
        subset["review"]
        .astype(str)
        .str.len()
    )

    subset = subset[
        subset["review_length"] > 120
    ]

    subset = subset.sort_values(
        ["usefulCount", "rating"],
        ascending=False
    )

    subset = subset.drop_duplicates(
        subset=["review"]
    )

    return subset[
        [
            "drugName",
            "rating",
            "usefulCount",
            "review"
        ]
    ].head(3)


# HEADER
st.markdown(
    """
    <div class="main-title">
        🩺 Drug Recommendation System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        AI-powered medical condition prediction
        and drug recommendation system
    </div>
    """,
    unsafe_allow_html=True
)

# INPUT
symptoms = st.text_area(
    "Describe symptoms or feelings",
    height=160,
    placeholder=
    "Example:\n"
    "• severe headache and dizziness\n"
    "• feeling hopeless and tired lately\n"
    "• excessive thirst and frequent urination"
)

predict_btn = st.button(
    "Predict Condition",
    use_container_width=True
)

# ACTION
if predict_btn:

    if not is_valid_input(symptoms):

        st.warning(
            "Please enter more detailed medical symptoms."
        )

    else:

        condition, confidence, proba = predict(
            symptoms
        )

        if confidence < CONFIDENCE_THRESHOLD:

            st.warning(
                "Prediction confidence is low. "
                "Try describing symptoms more clearly."
            )

            st.caption(
                f"Confidence: {confidence:.2%}"
            )

        else:

            # PREDICTION CARD
            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.subheader(
                "Prediction"
            )

            st.success(
                f"Predicted Condition: {condition}"
            )

            st.write(
                "Confidence Level"
            )

            st.progress(confidence)

            st.caption(
                f"{confidence:.2%}"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

            # RECOMMENDED DRUGS
            st.subheader(
                "Recommended Drugs"
            )

            st.caption(
                "Recommendations are generated "
                "from highly rated patient reviews."
            )

            recommendations = get_recommendations(
                condition
            )

            if len(recommendations) == 0:

                st.info(
                    "No recommendations available."
                )

            else:

                for drug in recommendations.index:

                    with st.expander(
                        f"💊 {drug}"
                    ):

                        st.markdown(
                            f"""
                            <div class="drug-card">
                                {drug}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            "### Patient Reviews"
                        )

                        reviews = get_drug_reviews(
                            drug
                        )

                        if len(reviews) == 0:

                            st.info(
                                "No reviews available."
                            )

                        else:

                            for _, row in reviews.iterrows():

                                st.markdown(
                                    f"""
                                    <div class="card">

                                    <b>⭐ Rating:</b>
                                    {row['rating']}

                                    &nbsp;&nbsp;&nbsp;

                                    <b>👍 Useful:</b>
                                    {row['usefulCount']}

                                    <br><br>

                                    <div class="review-box">
                                    "{row['review']}"
                                    </div>

                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

# FOOTER
st.markdown("---")

st.caption(
    "Drug Recommendation System is a machine "
    "learning application designed to predict medical "
    "conditions based on user-described symptoms and "
)