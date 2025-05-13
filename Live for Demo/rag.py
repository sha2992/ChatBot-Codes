import os
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
load_dotenv(find_dotenv())

from langchain_deepseek import ChatDeepSeek

api = os.environ.get("DEEPSEEK_API_KEY")

llm = ChatDeepSeek(
    model="google/gemini-2.0-flash-001",
    api_key=api,
    api_base="https://api.deepinfra.com/v1/openai",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

@st.cache_resource
def load_file(file_path):
    loader = TextLoader(file_path=file_path
                    , autodetect_encoding=True)

    docs = loader.load()

    return docs

file_path = "DiabetesCheck.txt"

docs = load_file(file_path)

def create_chunks(extracted_documents):
    text_spliter = CharacterTextSplitter(chunk_size=1300
                                         , chunk_overlap=450
                                         , separator = '\n\n\n\n'
                                         )
    
    text_chunks = text_spliter.split_documents(extracted_documents)
    return text_chunks

text_chunks = create_chunks(docs)

#Vectore Embeddings

def embedding_model():
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    return embedding_model

embedding_model = embedding_model()

#Store embedding vector
db = FAISS.from_documents(
    documents=text_chunks
    , embedding=embedding_model
    )

retriever = db.as_retriever(search_kwargs={"k":10}) 
   
### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

### Answer question ###
qa_system_prompt = """Act like an expert on diabetec consultant.
        You will be given questions in English, Bangla, or Banglish about diabete issues or suggestions or information. Your job is to answer these questions strictly based on the provided context.

        ### Instructions:
        Organize and analyze all the given contexts before answering. Name of the context is 'বাংলাদেশ ডায়াবেটিক এসোসিয়েশন এর তথ্য' or you can use 'বাংলাদেশ ডায়াবেটিক এসোসিয়েশন এর নির্দেশনা'. So If you need to structure a sentence like 'according to the Context' or 'in the Context' you should say 'বাংলাদেশ ডায়াবেটিক এসোসিয়েশন এর তথ্য মতে' or 'বাংলাদেশ ডায়াবেটিক এসোসিয়েশন এর নির্দেশনা অনুযায়ী'.
        
        2. Provide answers in Bangla.
        
        3. Name of the context is 'বাংলাদেশ ডায়াবেটিক এসোসিয়েশন এর তথ্য'. 
        
        4. Keep your answers concise.
        
        5. If you do not specifically answer a question or further information needs to answer the question you can ask what additional information you need to answer the question.
                - For example if anyone want to know whether he has a chance of being diabete, you should able to ask him questions that is related to the context of diabetes possibility.
                - If anyone says about his health situation you can ask him to do some task based on his query and relevence from context.         
        
        6. If a user asks for a height that is not present in the context, simply state that it is not available without making any assumptions or providing an estimated answer. You only have even number of heights within a certain range, for odd number you can not suggest.
        
        7. If the user don't specifically tell about 'রোজা', 'রমজান', 'ইফতার', 'fasting', then your response should not return the response with 'ইফতার' or 'সেহেরী' rather you would tell about normal day, not ramadan day.
        
        8. Until user asks no need to say the source of your answer, like answer should not contain 'খাদ্য তালিকা ৬', 'খাদ্য তালিকা ৭' and similar. If from context you get these you will not show that to your answer until user specifically asked for it.
        
        9. If a requested calorie-based diet chart is unavailable in the context, simply state that it is not available without generating a new one.
        
        10. If context does not have relevant information, 
                    - Understand whether it is relevant to diabetes and any term from the context or not
                          - the the term is relevant or from the context, say that your context don't provide you information, but you are giving some concise information from your global knowledge. 
                          - if the term or question is not relevant to the context or diabetes, say that you don't have information about that.

        11. After getting information from context you would answer the question in a way that user can find it friendly to read. Write in a way that even a kid of 10 years old can understand.
        
        12. After providing asked question's answer you can do the following things upon situation: 
                    - ask him whether he wants to know more or any other help you should do when you can answer the question directly from context.
                    - ask him to do relevant task say context said for any question that you could not answer specificly because of precuisit information, ask him to do that task (or tasks) and let you know.
        
        13. If user asked about any of his syndrome, you should understand that and match that with the relevant portion from the context and give him suggestions and can ask him to do some tasks (like check your glucose level or bp level) and notify you so you can suggest some more.
        
        14. For some query you may need to ask about the gender of the user along with the relevent question from context. For example: If question is about asking about someone's weight you need to ask his gender along with other relevant question to answer specificly.  
        
        15. If user ask about whether he can eat some food or how much he can eat certain food, you will check the food details from the context whether that food listed or not and how much he can eat that food in a day, and if any specific time is mentioned for eating say that to the user.
        ### Important:
        - Do not provide information outside the given context.
        - If the context does not contain relevant information, state that explicitly in Bangla.
        - You will never show from which portion of the context you got the answer.
        - You will never say like 'according to context' or 'context অনুযায়ী' or 'context' or 'কন্টেক্সট' instead of that you would say 'according to BADAS Book' or 'BADAS Book অনুযায়ী' or 'BADAS Book'.
    
        {context}
        """
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def answer_query(query):
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    ans = conversational_rag_chain.invoke(
                {"input": query}
                , config={ "configurable": {"session_id": "abc123"} },
            )['answer']
    
    return ans

def ans_rep(response):
    tables = {
        "সবজি (তালিকা ক)": ["ফুলকপি, বাঁধাকপি, মুলা, ওলকপি, ক্যাপসিকাম, কাঁচা টমেটো, কাঁচা পেঁপে, শসা, খিরা, উচ্ছে, করলা, ঝিংগা, চিচিংগা, পটল, চালকুমড়া, ডাঁটা, লাউ, সাজনা, ধুন্দল ও কাঁচা মরিচ"],
        "সবজি (তালিকা খ)": ["আলু", "মিষ্টি কুমড়া", "কচু", "থোড়", "বিট", "কাঁচা কলা", "বরবটি", "মোচা", "সিম", "গাজর", "কাঁকরোল", "শালগম", "ইঁচড়", "ঢেড়স", "বেগুন", "পাকা টমেটো", "কচুমূখী", "পঞ্চমূখী"],
        "ফল (তালিকা ক)": ["কাঁচা আম", "কাঁচা পেয়ারা", "কচি ডাবের পানি", "চালতা", "জামরুল", "সবুজ বড়ই", "বাঙ্গি", "তেতুল", "কাল জাম", "পানিফল", "জলপাই", "জাম্বুরা", "আমড়া", "আমলকি"],
        "ফল (তালিকা খ)": ['পাকা পেয়ারা','আম','মালটা অথবা কমলা','কাঁঠাল','শরিফা','সফেদা','আনারস','পাকা পেঁপে','তরমুজ','মিষ্টি বরই','বেদানা','তাল','লিচু','কেশর','দেশী খেজুর','পাকা বেল','খোরমা']
    }

    for key, value_list in tables.items():
        if key in response:
            response = response.replace(key, ', '.join(value_list))

    return response