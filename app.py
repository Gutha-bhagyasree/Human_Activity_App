import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = joblib.load('har_model.pkl')

# Streamlit UI
st.title("📱 Human Activity Recognition via Smartphone")
st.write("Upload a CSV file containing your sensor data to predict the activity.")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the CSV
        data = pd.read_csv(uploaded_file)
        if data.shape[1] != 561:
            data = pd.read_csv(uploaded_file, header=None)

        st.subheader("🧾 Uploaded Data Sample")
        st.dataframe(data.head())

        # Model prediction
        prediction = model.predict(data)

        # Map numeric labels to activity names
        activity_map = {
            1: "WALKING",
            2: "WALKING_UPSTAIRS",
            3: "WALKING_DOWNSTAIRS",
            4: "SITTING",
            5: "STANDING",
            6: "LAYING"
        }
        prediction_labels = [activity_map[p] for p in prediction]

        st.success("✅ Prediction Completed!")

        # Display results
        if len(prediction_labels) == 1:
            st.subheader(f"🎯 Predicted Activity: *{prediction_labels[0]}*")
        else:
            st.subheader("🎯 Predicted Activities:")
            st.write(prediction_labels)

        # Visualization: Bar chart
        activity_counts = pd.Series(prediction_labels).value_counts()
        st.subheader("📊 Activity Distribution")
        st.bar_chart(activity_counts)

        # Visualization: Pie chart
        st.subheader("📈 Activity Proportion")
        fig, ax = plt.subplots()
        ax.pie(activity_counts, labels=activity_counts.index, autopct="%1.1f%%", startangle=90)
        ax.set_title("Activity Proportion")
        st.pyplot(fig)

        # Visualization: Heatmap (top 20 features)
        st.subheader("🔥 Top 20 Sensor Correlations")
        top_features = data.var().sort_values(ascending=False).head(20).index
        corr = data[top_features].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠ Error: {str(e)}")