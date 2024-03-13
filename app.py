from dotenv import load_dotenv
import streamlit as st
from langchain_experimental.agents import create_csv_agent
from langchain.llms import HuggingFaceHub
from htmlTemplate import css, bot_template, user_template

load_dotenv()

def main():
    st.set_page_config(
        page_title= "QnA with CSV",
        page_icon=":robot_face:",
        layout="wide"
    )
    st.write(css, unsafe_allow_html=True)

    st.title("QnA with CSV 🤖")
    user_csv = st.file_uploader(label = "", type= "csv")
    if not user_csv:
        st.warning("Please Upload your CSV file 🚨")

    llm = HuggingFaceHub(
            repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
            model_kwargs = {
                'max_new_tokens': 249, 
                'temperature': 0.3,
            }
        )
    user_input = st.chat_input("Ask your Question about CSV files",
                               disabled= not user_csv)
    
    with st.sidebar:
        st.subheader("Example Questions:")
        example_questions = [
            "What is the total number of rows in the CSV?",
            "Can you show me the first 5 rows of the CSV?",
            "What are the column names in the CSV?",
            "How many columns does the CSV have?",
            "What is the data type of a specific column in the CSV?",
            "Can you provide a summary statistics for the numerical columns?",
            "Are there any missing values in the CSV?",
            "Can you filter the data based on a specific condition?",
            "What is the average value of a numerical column?",
        ]

        selected_example = st.selectbox("Select an example question:", example_questions, disabled= not user_csv)

        if not user_csv:
            st.warning("Please Upload your CSV file 🚨")
        
        if st.button("Use Selected Example"):
            user_input = selected_example
    
    if user_csv is not None:
        agent = create_csv_agent(
            llm = llm,
            path = user_csv,
            verbose = True,
            handle_parsing_errors=True
        )

        if user_input is not None and user_input != "":
            st.write(user_template.replace("{{MSG}}",user_input), unsafe_allow_html= True)         
            with st.spinner("Processing..."):   
                response = agent.run(user_input)
                st.write(bot_template.replace("{{MSG}}",response), unsafe_allow_html= True)

if __name__ == "__main__":
    main()