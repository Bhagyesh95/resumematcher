import streamlit as st
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_model():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        st.write("Model Loaded sucessfully")
        return model
    except Exception as e:
        st.error("Failed to load model: {e}")
        return None
model = load_model()



st.title("AI Resume Optimizer")

resume = st.text_area("Paste Your Resume Text", height=200)
job_desc = st.text_area("Paste the Job Description", height=200)

if st.button("Analyze"):
    if not resume or not job_desc:
        st.warning("Please paste both resume and job description")
    else:
        resume_embedding = model.encode(resume, convert_to_Tensor=True)
        job_embedding = model.encode(job_desc, convert_to_Tensor=True)

        similarity = util.pytorch_cos_sim(resume_embedding, job_embedding).item()
        similarity_score = round(similarity * 100, 2)
        st.markdown("### Match Score: {}%".format(similarity_score))

        resume_words = set(resume.lower().split())
        job_words = set(job_desc.lower().split())
        missing_words = job_words - resume_words

        st.markdown("Suggested Improvements:")
        if missing_words:
            st.write("Considere adding these words to match job Better")
            st.write(",".join(list(missing_words)[:20]))
        else:
            st.write("Your Resume already covers most Key Points!")
