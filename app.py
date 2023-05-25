import streamlit as st
# from streamlit_extras.add_vertical_space import add_vertical_space
import pickle


from dotenv import load_dotenv
import os
from langchain import ConversationChain 
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory



#sidebar contents

with st.sidebar:
    st.title("LLM Chat App")
    st.markdown(
        """
        Streamlit app"""
    )
   
    st.write("## Chatbot")

def main():
    st.header("Chat with Your PDF ")
    load_dotenv()
    pdf=st.file_uploader("Upload your PDF file",type=['pdf'])
   

    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()

        text_splitter=CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunks=text_splitter.split_text(text=text)
   
   #Embeddings
        embeddings=OpenAIEmbeddings()
        VectorStore=FAISS.from_texts(chunks,embedding=embeddings)
        store_name=pdf.name[:-4]

        if os.path.exists(f"{store_name}.pk1"):
            with open(f"{store_name}.pk1","rb") as f:
                VectorStore=pickle.load(f)
           
        else:
             embeddings=OpenAIEmbeddings()
             with open(f"{store_name}.pk1","wb") as f:
                 pickle.dump(VectorStore,f)


# Accept user questions/queries
        query=st.text_input("Ask your question")
        if query:
            docs=VectorStore.similarity_search(query=query,k=3)
            # llm = OpenAI(temperature=0.7, max_tokens=100)
            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response=chain.run(input_documents=docs,question=query)
            # with get_openai_callback() as cb:
            #     print(cb)


            # Calucluate Cost
            with get_openai_callback() as cb:
                response=chain.run(input_documents=docs,question=query)
                print(cb)

            # Display Response
                st.write(response)

                # Get Youtube Link
                button=st.button("Get Youtube Link that relates to the response")
                if button:
                    llm =OpenAI(model_name="text-davinci-003")
                    prompt=f"Provide a Youtube link that relates to {response}"
                    st.write(f"Here is a Youtube Link that relates to {llm(prompt)}") 






 



        


 

if __name__ == "__main__":
    main()


