import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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
import psycopg
from datetime import datetime
from langchain.chains import LLMChain
import json
from langchain_deepseek import ChatDeepSeek
from deep_translator import GoogleTranslator


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

#### User Info ############
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

def get_user_info(user_id):
    conn = psycopg.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()

    ######################## Query to get the value of information of a particular user #######################
    cursor.execute(
        """WITH ranked_glucose AS (
                SELECT 
                    user_id, 
                    value, 
                    context,
                    reading_time::date AS reading_date, 
    	            reading_time::time AS reading_times,
                    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY reading_time DESC) AS glucose_rank
                FROM glucose_readings
            ),
            pivoted_glucose AS (
                SELECT
                    CAST(user_id AS INTEGER) user_id,
                    MAX(CASE WHEN glucose_rank = 1 THEN value END) AS last_recorded_glucose,
                    MAX(CASE WHEN glucose_rank = 1 THEN context END) AS last_recorded_glucose_context,
                    MAX(CASE WHEN glucose_rank = 1 THEN reading_date END) AS last_recorded_glucose_date,
                    MAX(CASE WHEN glucose_rank = 1 THEN reading_times END) AS last_recorded_glucose_time,
                    MAX(CASE WHEN glucose_rank = 2 THEN value END) AS second_last_recorded_glucose,
                    MAX(CASE WHEN glucose_rank = 2 THEN context END) AS second_last_recorded_glucose_context,
                    MAX(CASE WHEN glucose_rank = 2 THEN reading_date END) AS second_last_recorded_glucose_date,
                    MAX(CASE WHEN glucose_rank = 2 THEN reading_times END) AS second_last_recorded_glucose_time
                FROM ranked_glucose
                GROUP BY user_id
            )
            SELECT 
                ud."first_name",ud.dob, EXTRACT(YEAR FROM age(current_date, dob)) AS age,
                ud.gender,ud.height_cm,ud.weight_kg,
                last_recorded_glucose, last_recorded_glucose_context, last_recorded_glucose_date, last_recorded_glucose_time,
                second_last_recorded_glucose, second_last_recorded_glucose_context, second_last_recorded_glucose_date, second_last_recorded_glucose_time
            FROM users ud 
                left join pivoted_glucose pg on ud.id=pg.user_id  
            WHERE id = %s""", (user_id,)) 
    result = cursor.fetchone()
    cursor.close()
    conn.close()

########################## Parsing the value from the query ##############################
    if result:
        return {
            "name": result[0],
            "dob": str(result[1]),
            "age": result[2],
            "gender": result[3],
            "height": result[4],
            "weight": result[5],
            "last_glucose":  result[6],
            "last_glucose_context": result[7],
            "last_glucose_date": str(result[8]),
            "last_glucose_time": str(result[9]),
            "2last_glucose":  result[10],
            "2last_glucose_context": result[11],
            "2last_glucose_date": str(result[12]),
            "2last_glucose_time": str(result[13])
        }
    else:
        return f"""no profile found"""

########################### calling earlier function to get user information  ########################## 

user_id = 1  ################# for the time being using static user ###############
info=get_user_info(user_id) 


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

# Step 2: Define QA Prompt with User Info (Template Approach)
qa_system_prompt = """
Act like an expert on diabetic consultation.
You will be given questions in English, Bangla, or Banglish about diabetes issues, suggestions, or information. 
Your job is to answer these questions strictly based on the provided context.

You should remember the user information. The user information is as follows:

The user is {name}, their age is {age}, gender is {gender}, height is {height}, weight is {weight}. 
Their some of the last recorded glucose level also in the info, last glucose level {last_glucose} recorded as {last_glucose_context} in {last_glucose_date} at {last_glucose_time}. Second last glucose level {last2_glucose} recorded as {last2_glucose_context} in {last2_glucose_date} at {last2_glucose_time}.

While using time value, make sure you can understand what moment of the day and deliver in 12 o clock. Like 20:00 should be 8 PM, 8:30 should be 8:30 AM.
### Instructions:
1. Organize and analyze all the given contexts before answering.
2. Provide answers in Bangla.
3. Reference the source of the information from the context in your answer.
4. Keep your answers concise.
5. If you do not specifically answer a question or further information needs to answer the question you can ask what additional information you need to answer the question.
7. For Height, if the asked height is not present in the context just say that but don't give any assumed answer.
8. If the user don't specifically tell about 'রোজা', 'রমজান', 'ইফতার', 'fasting', then your response should not return the response with 'ইফতার' or 'সেহেরী' rather you would tell about normal day, not ramadan day.
9. Until user asks no need to say the source of your answer.
10. If a requested calorie-based diet chart is unavailable in the context, simply state that it is not available without generating a new one.
11. After getting information from context you would answer the question in a way that user can find it friendly to read. Write in a way that even a kid of 10 years old can understand.
        
12. After providing asked question's answer you can do the following things upon situation: 
        - ask him whether he wants to know more or any other help you should do when you can answer the question directly from context.
        - ask him to do relevant task say context said for any question that you could not answer specificly because of precuisit information, ask him to do that task (or tasks) and let you know.
        
13. If user asked about any of his syndrome, you should understand that and match that with the relevant portion from the context and give him suggestions and can ask him to do some tasks (like check your glucose level or bp level) and notify you so you can suggest some more.
        
14. For some query you may need to ask about the gender of the user along with the relevent question from context. For example: If question is about asking about someone's weight you need to ask his gender along with other relevant question to answer specificly.  
        
15. If user ask about whether he can eat some food or how much he can eat certain food, you will check the food details from the context whether that food listed or not and how much he can eat that food in a day, and if any specific time is mentioned for eating say that to the user.
16. When you get user's information like glucose reading, blood pressure reading, you should check that reading with the context whether he needs to follow any specical instruction or precausion according to the context. If so, state that in suggestive manner.
        - Remember that user can refer glucose level as diabetes level, blood pressure as bp or 'রক্তচাপ'. Keep check for any particular reading he is looking for like 'last' or 'শেষ' or 'সর্বশেষ' 
        
17. If user greet you, you will greet him back, if you already know his name from the context, greet him by stating his name.
### Important:
- Do not provide information outside the given context.
- If the context does not contain relevant information, state that explicitly in Bangla.
- You will never show from which portion of the context you got the answer.
- You will never say like 'according to context' or 'context অনুযায়ী' or 'context' or 'কন্টেক্সট' instead of that you would say 'বাংলাদেশ ডায়াবেটিক এসোসিয়েশন এর তথ্য মতে' or 'বাংলাদেশ ডায়াবেটিক এসোসিয়েশন এর নির্দেশনা অনুযায়ী'.
- If user ask about their information stored in the databse, just say that much they ask not the entire information you.
- All information should be in Bengali, like date or other informations, can use english for unit when necessary.
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt.partial(
            name=info["name"],
            dob=info["dob"],
            age=info["age"],
            gender=info["gender"],
            height=info["height"],
            weight=info["weight"],
            last_glucose = info['last_glucose'],
            last_glucose_context=info["last_glucose_context"],
            last_glucose_date = info['last_glucose_date'],
            last_glucose_time = info['last_glucose_time'],
            last2_glucose = info['2last_glucose'],
            last2_glucose_context=info["2last_glucose_context"],
            last2_glucose_date = info['2last_glucose_date'],
            last2_glucose_time = info['2last_glucose_time']
        )
    )


###################### Action #################################

######################## This is for checking whether any action in the intention ################
action_check_prompt = """
You are given a response to a user query. 
Your job is to determine whether any further action needs to be taken based on the content of the response.
the response you got can be in english, bengli or banglish, understand the language.

### Instructions:
1. Analyze the provided response carefully.
2. If the response indicates that further action is required
3. You may find instruction to do some action, action can be implicit or explicit
        - implicit action: there can be user information (name, address, birthdate, height, weight) in the response, when you those, you will save these information to database. Not need to get explicit instruction from response to save, you just save. If get instruction explicitly, then tell that you have saved this information.
        - explicit action: you may get instruction like set a reminder, alarm. You will get that time, and task to the database. You will set the data in the following format
        - graph action: you may get instruction like what was my last 6 months glucose level, give me last 6 months glucose level track. 
4. If action required, pass two variables, status with value yes and type implicit or explicit or graph depending on which type of action, if no further action is required, just pass variable status with no, null or empty to send type.
5. Keep your answer concise.

Response:
{query}
"""

action_check_chain = ChatPromptTemplate.from_messages(
    [
        ("system", action_check_prompt),
        ("human", "{query}"),
    ]
)

action_check_llm_chain = LLMChain(
    llm=llm,
    prompt=action_check_chain
)


############## to deal implicit action (from natural language) #####################

implicit_action ="""
You will get implicit action type data, in Bengali, English or Banglish. From implicit action you will get information regarding users.
User's information can be their name, age, date of birth, height, weight, gender. Depending on the query you get you will provide several variables:
    Variable_name
        (You can find "name", "age","height","gender" these would be variable name.)
    Variable_value
        (the value user tells corresponding to a variable will be its value.)

    example: query says My height is 176 cm. You will set height at Variable_name and 176 cm at Variable_value. 

    you may find query containing multiple values like My age is 30 and height it 176 cm. you will set age, height (seperating by comma) at Variable_name and 30, 176 cm (seperating by comma) at Variable_value. No need to take the unit of age.

    Response:
    {query}
"""

implicit_action_data = ChatPromptTemplate.from_messages(
    [
        ("system",implicit_action),
        ("human","{query}"),
    ]
)

implicit_action_data_llm_chain = LLMChain(
    llm=llm,
    prompt=implicit_action_data
)


######################## to deal explicit action (reminder type) ###########################
action_data = """
You will get the action query. Query can be in bangla, english, banglish. You will need to store the data of the query two variables:

    - Action_name 
        (you can find "reminder", "mone koriye deben", "মনে করিয়ে দেবেন", these will be named as Reminder )
    - Action_time
        (you may find bikal 5 ta, that means 5 PM, for sokal ba vor 5 ta, it will be 5 AM)
    example: query says set reminder at 5 PM. You will set Reminder at Action_name and 5 PM at Action_time.
                            bikal 5 tay reminder deben. You will Reminder at Action_name and 5 PM at Action_time.

Response:
{query}
"""

action_data_chain = ChatPromptTemplate.from_messages(
    [
        ("system", action_data),
        ("human", "{query}"),
    ]
)

action_data_llm_chain = LLMChain(
    llm=llm,
    prompt=action_data_chain
)


###################################### graph action ###############################

graph_chain_data = """
You will get graph action type in english, bengli or banglish, understand the language.

Your job is to determine what table and sql query and time frame using the response.
    Table_name
        - determine the table the response refer
            - if glucose or diabetes or glucometer reading in the query then glucose_readings
            - if bp or blood pressure then bp_readings
    interval
        - determine from the response that what is the end date it referring
            -for example 3 months earlier means '3 months'
Response:
{query}
"""

graph_chain_llm = ChatPromptTemplate.from_messages(
    [
        ("system",graph_chain_data),
        ("human","{query}"),
    ]
)

graph_action_data_llm_chain = LLMChain(
    llm=llm,
    prompt=graph_chain_llm
)

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

##################### to replace veg or fruits name #######################
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


########################## final output checking status ##############################
def output(query):
    ############# stripping the value for next test ################
    action_state = json.loads(action_check_llm_chain.invoke(query)['text'].replace("```json", "").replace("```", "").strip())

    if action_state['status'] == 'yes':
        if action_state['type']=='explicit':

            #### stripping value to store in db and conversation #########
            raw_output = action_data_llm_chain.invoke(query)['text']
            cleaned = raw_output.replace("```json", "").replace("```", "").strip()

            # Parse the cleaned string into a dictionary
            data = json.loads(cleaned)
        
            conn = psycopg.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )
            cursor = conn.cursor()
            cursor.execute("INSERT INTO reminders (user_id, reminder_name, reminder_time) VALUES (%s, %s, %s)", 
                            (1, data['Action_name'], datetime.strptime(data['Action_time'], "%I %p").time().strftime("%H:%M:%S")))
            conn.commit()
            conn.close()

            ex = f"Sure, {data['Action_name']} will be done at {data['Action_time']}"

            ############## translating for the conversation ##############
            output = GoogleTranslator(source='en', target='bn').translate(ex)

        ################ For Dealing with implicit actions from natural conversation ###############
        elif action_state['type']=='implicit':
            
            #### stripping value to store in db and conversation #########

            raw_output = implicit_action_data_llm_chain.invoke(query)['text']
            cleaned = raw_output.strip().split('\n')

            data = {}

            for line in cleaned:
                key, value = line.split(':', 1)
                data[key.strip()] = value.strip()

            ########### Currently not storing it in db ################
            # conn = psycopg.connect(
            #     dbname=DB_NAME,
            #     user=DB_USER,
            #     password=DB_PASSWORD,
            #     host=DB_HOST,
            #     port=DB_PORT
            # )
            # cursor = conn.cursor()
            # cursor.execute("
            ### Update table ###
            im = f"your {data['Variable_name']} is {data['Variable_value']}, should i store it for further use?"
            
            output = GoogleTranslator(source='en', target='bn').translate(im)

        ################ For Dealing with Data of Graph ##################
        elif action_state['type']=='graph':
            data = json.loads(graph_action_data_llm_chain.invoke(query)['text'].replace("```json", "").replace("```", "").strip())

            conn = psycopg.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )
            cursor = conn.cursor()
            query = psycopg.sql.SQL("""
                SELECT reading_time::date, value
                FROM {table}
                WHERE user_id = %s AND
                      reading_time >= (CURRENT_DATE - (%s)::interval) AND
                      reading_time < (CURRENT_DATE + INTERVAL '1 day')
            """).format(
                table=psycopg.sql.Identifier(data['table_name'])
            )

            cursor.execute(query, (1, data['interval']))

            result = cursor.fetchall()
            conn.commit()
            conn.close()

    else:
        output = answer_query(query)

    return output