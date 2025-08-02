import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

# Load Data
df = pd.read_csv("CareerAdvisory.csv")
df.columns = df.columns.str.strip()

st.set_page_config(page_title="🎓 Career Advisory Predictor", layout="wide")
st.title("🎓 Career Advisory Predictor (Post-12th)")

# PDF Export Option
st.markdown("## 🖨 Export as PDF")
show_all = st.checkbox("Show All Sections for PDF Export", value=False) 

# ================= Tabs =====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 Bar Chart", "📈 Line Chart", "🌞 Sunburst", "🔘 Scatter",
    "📌 Heatmap", "📚 Histogram", "🎯 Predictor", "🍩 Donut Chart", "🧠 Extra Advisory Details" ," 📅Application Tracker Form"
])

# ============== TAB 1: Bar Chart ==================
def tab1_content():
    st.subheader("📈 Aptitude Score vs Target Career")

    selected_stream = st.selectbox("🎓 Select Stream", df["Stream_in_12th"].unique())

    # Filter dataset by selected stream
    stream_df = df[df["Stream_in_12th"] == selected_stream]

    # Get top 10 career interests by frequency
    top_careers = stream_df["Career_Interest"].value_counts().nlargest(10).index.tolist()
    filtered_df = stream_df[stream_df["Career_Interest"].isin(top_careers)]

    # Group by Career Interest and calculate average aptitude score
    grouped_df = filtered_df.groupby("Career_Interest")["Aptitude_Test_Score"].mean().reset_index()

    # Plot
    fig = px.bar(
        grouped_df,
        x="Career_Interest",
        y="Aptitude_Test_Score",
        color="Career_Interest",
        title=f"🎯 Average Aptitude Score for Top 10 Careers in {selected_stream}",
        labels={"Aptitude_Test_Score": "Avg. Aptitude Score"},
        color_discrete_sequence=px.colors.qualitative.Vivid
    )

    fig.update_layout(
        xaxis_title="Career Interest",
        yaxis_title="Average Aptitude Score",
        xaxis_tickangle=-30,
        height=600,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

# ============== TAB 2: Line Chart ==================
def tab2_content():
    st.subheader("📈 Average 12th Percentage by Target Job Role")

    # Group and sort by average 12th percentage
    avg_target_job = df.groupby("Target_Job_Role")["12th_Percentage"].mean().sort_values(ascending=False).reset_index()

    # Plot a single joined line (no color per category)
    fig = px.line(
        avg_target_job,
        x="Target_Job_Role",
        y="12th_Percentage",
        markers=True,
        title="🎯 Average 12th Percentage vs Target Job Role",
        color_discrete_sequence=["#FF6347"]  
    )

    # Layout adjustments
    fig.update_layout(
        xaxis_title="Target Job Role",
        yaxis_title="Average 12th Percentage",
        xaxis_tickangle=45,
        height=600,
        template="plotly_white",
        showlegend=False
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# ============== TAB 3: Sunburst ==================
def tab3_content():
    st.subheader("🌞 Stream → Career Interest → Target Job Role Breakdown")

    try:
        # Top 10 career interests per stream
        top_careers = df.groupby("Stream_in_12th")["Career_Interest"].value_counts()\
            .groupby(level=0).nlargest(10).reset_index(level=0, drop=True).reset_index(name="Count")

        # Filter df for these top careers
        sunburst_df = df.set_index(["Stream_in_12th", "Career_Interest"]).loc[
            top_careers.set_index(["Stream_in_12th", "Career_Interest"]).index
        ].reset_index()

        # Create sunburst chart with additional path
        fig = px.sunburst(
            sunburst_df,
            path=["Stream_in_12th", "Career_Interest", "Target_Job_Role"],
            color="Career_Interest",
            color_discrete_sequence=px.colors.qualitative.Bold,
            maxdepth=3
        )

        # 🔸 Increase size
        fig.update_layout(
            height=800,
            margin=dict(t=50, l=0, r=0, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error generating sunburst: {e}")

# ============== TAB 4: Scatter Plot ==================
def tab4_content():
    st.subheader("🔘 Aptitude vs Percentage by Interest Domain")

    fig = px.scatter(
        df,
        x="12th_Percentage",
        y="Aptitude_Test_Score",
        color="Interest_Domain",
        size="Maths_Score",
        text="12th_Percentage",
        title="🎯 Aptitude vs Percentage (with Maths Score Bubble Size)",
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    fig.update_traces(
        textposition='middle center', 
        marker=dict(
            opacity=0.6,  
            line=dict(width=1.5, color='black')  
        )
    )

    fig.update_layout(
        xaxis_title="12th Percentage",
        yaxis_title="Aptitude Test Score",
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)

# ============== TAB 5: Heatmap ==================
def tab5_content():
    st.subheader("📌 Correlation Heatmap")
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="Spectral", fmt=".2f", ax=ax)
    st.pyplot(fig)

# ============== TAB 6: Histogram ==================
def tab6_content():
    st.subheader("📚 Score Distribution by Career Domain")

    # Dropdowns for subject and stream selection
    col = st.selectbox("Choose Score Type", ["Maths_Score", "Physics_Score", "Biology_Score", "English_Score"])
    stream_choice = st.selectbox("Choose Stream", df["Stream_in_12th"].unique())

    # Filter the dataframe based on selected stream
    filtered_df = df[df["Stream_in_12th"] == stream_choice]

    # Create histogram by Career Domain
    fig = px.histogram(
        filtered_df,
        x=col,
        color="Interest_Domain",  
        nbins=20,
        title=f"{col} Distribution for {stream_choice} by Career Domain",
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    fig.update_layout(
        xaxis_title=col.replace("_", " "),
        yaxis_title="Student Count",
        barmode="overlay",
        bargap=0.1,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

# ============== TAB 7: ML Predictor ==================
def tab7_content():
   
    import pickle

    # Load models
    with open("models/Target_College.pkl", "rb") as f:
        model_college = pickle.load(f)

    with open("models/Career_Interest.pkl", "rb") as f:
        model_career = pickle.load(f)

    with open("models/Backup_Course.pkl", "rb") as f:
        model_backup = pickle.load(f)

    with open("models/Counselor_Recommendation.pkl", "rb") as f:
        model_counsel = pickle.load(f)

    with open("models/Target_Job_Role.pkl", "rb") as f:
        model_jobrole = pickle.load(f)

    with open("models/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)



# ============== TAB 8: Donut Charts ==================
def tab8_content():
    st.subheader("🍩 Stream-wise Career Interest Donuts")

    for stream in ["Commerce", "Science", "Arts"]:
        st.markdown(f"#### {stream} Stream")
        sub_df = df[df["Stream_in_12th"] == stream]
        donut_data = sub_df["Career_Interest"].value_counts().nlargest(10).reset_index()
        donut_data.columns = ["Career_Interest", "Count"]
        fig = px.pie(
            donut_data, values="Count", names="Career_Interest",
            hole=0.5, title=f"{stream} Career Interests",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig, use_container_width=True)
        
# ============== TAB 9: Extra Advisory Details ==================
def tab9_content():
    st.subheader("🧾 Advanced Career Expansion Form")
    with st.form("expansion_form"):
        st.markdown("### 📚 Academic Interests")
        academic_goals = st.multiselect("Which areas are you interested in?", [
            "Engineering", "Medical", "Management", "Design", "Humanities",
            "Pure Science", "Law", "Commerce", "Vocational"
        ])
        study_style = st.radio("Preferred Study Style:", [
            "Theoretical", "Practical", "Mixed"], horizontal=True)

        st.markdown("### 🌐 Study Abroad Planner")
        study_abroad = st.toggle("Interested in Studying Abroad?")
        if study_abroad:
            country = st.selectbox("Target Country", [
                "USA", "UK", "Canada", "Australia", "Germany", "Other"])
            language_test = st.radio("Language Test Taken?", ["IELTS", "TOEFL", "None"], horizontal=True)

        st.markdown("### 🧠 Psychological Profile")
        motivation_level = st.slider("Motivation Level (1-10)", 1, 10, 5)
        stress_handling = st.selectbox("How do you handle stress?", [
            "Calmly", "Productively", "Need support", "Not well"])
        leadership = st.radio("Do you enjoy leading teams?", ["Yes", "Sometimes", "No"])

        st.markdown("### 💼 Career Inclination")
        industry_interest = st.multiselect("Which industries fascinate you most?", [
            "Tech", "Healthcare", "Finance", "Education", "Public Services",
            "Media", "Art & Design", "Research", "Entrepreneurship"])
        remote_ok = st.radio("Are you comfortable working remotely?", ["Yes", "No", "Hybrid"], horizontal=True)

        submit_expansion = st.form_submit_button("Submit Details")

    if submit_expansion:
        st.success("🎯 Your profile has been recorded!")
        st.info("📌 Personalized Guidance: Based on your academic, psychological and industry preferences, consider hybrid academic paths or interdisciplinary programs aligned with your goals. For in-depth analysis, reach out to a certified career counselor.")
        st.balloons()
        
# ========== TAB 10: 📅 Application Tracker ==========
def tab10_content():
    st.subheader("📅 Application Tracker")

    if "applications" not in st.session_state:
        st.session_state.applications = []

    st.markdown("### ➕ Add New Application")
    with st.form("app_form"):
        col1, col2 = st.columns(2)
        with col1:
            app_type = st.selectbox("Application Type", ["College", "Exam"])
            
            if app_type == "College":
                college_list = sorted(df["Target_College"].dropna().unique())
                name = st.selectbox("Select College", college_list)
            else:
                name = st.text_input("Name of Exam")

            status = st.radio("Status", ["Applied", "Pending", "Accepted", "Rejected"], horizontal=True)

        with col2:
            app_date = st.date_input("Application Date")
            remarks = st.text_input("Remarks (optional)")

        submitted = st.form_submit_button("Add Application")
        if submitted and name:
            st.session_state.applications.append({
                "Type": app_type,
                "Name": name,
                "Date": str(app_date),
                "Status": status,
                "Remarks": remarks
            })
            st.success("✅ Application added successfully!")

    st.markdown("---")
    st.markdown("### 📋 Your Applications")

    if st.session_state.applications:
        status_filter = st.selectbox("Filter by Status", ["All", "Applied", "Pending", "Accepted", "Rejected"])
        filtered_apps = st.session_state.applications

        if status_filter != "All":
            filtered_apps = [app for app in filtered_apps if app["Status"] == status_filter]

        df_apps = pd.DataFrame(filtered_apps)
        st.dataframe(df_apps, use_container_width=True)
    else:
        st.info("No applications added yet. Fill the form above to get started.")


# --- Show all sections or use tabs ---
if show_all:
    st.markdown("""
    <script>
        window.addEventListener('load', function() {
            window.print();
        });
    </script>
    """, unsafe_allow_html=True)

    tab1_content()
    tab2_content()
    tab3_content()
    tab4_content()
    tab5_content()
    tab6_content()
    tab7_content()
    tab8_content()
    tab9_content()
    tab10_content()
else:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📊 Bar Chart", "📈 Line Chart", "🌞 Sunburst", "🔘 Scatter",
        "📌 Heatmap", "📚 Histogram", "🎯 Predictor", "🍩 Donut Chart", "🧠 Extra Advisory Details", "📅 Tracker"
    ])
    with tab1: tab1_content()
    with tab2: tab2_content()
    with tab3: tab3_content()
    with tab4: tab4_content()
    with tab5: tab5_content()
    with tab6: tab6_content()
    with tab7: tab7_content()
    with tab8: tab8_content()
    with tab9: tab9_content()
    with tab10: tab10_content()

