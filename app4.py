# 以下を「app2.py」に書き込み
import streamlit as st
import openai
import os
from dotenv import load_dotenv
import platform
from pyngrok import ngrok
import chromadb
import langchain



from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader('まとめ2.pdf')
pages = loader.load_and_split()

load_dotenv()

# pipをアップグレード
os.system('/home/adminuser/venv/bin/python -m pip install --upgrade pip')

# 追加の開発パッケージをインストール
os.system('/home/adminuser/venv/bin/python -m pip install python3-dev')

# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key =  os.environ['OPENAI_API_KEY']

# openai.api_key = st.secrets.OpenAIAPI.openai_api_key


# st.session_stateを使いメッセージのやりとりを保存
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "あなたは優秀なアシスタントAIです。"}
        ]

llm = ChatOpenAI(temperature=0, model_name="gpt-4")

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(pages, embedding=embeddings, persist_directory=".")
vectorstore.persist()

# チャットボットとやりとりする関数
def communicate():
    # st.session_stateを初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    pdf_qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)
    messages = st.session_state["messages"]

    user_message = {"role": "user", "content": st.session_state["user_input"]}
    messages.append(user_message)

    response = openai.ChatCompletion.create(
         model="gpt-4",
         messages=messages,
         temperature=0.0
     )
    query = user_message["content"]  # ユーザーからの入力を質問として使用
    chat_history = []

    result = pdf_qa({"question": query, "chat_history": chat_history})

    bot_message = result["answer"]
    messages.append({"role": "assistant", "content": bot_message})

    # bot_message = response["choices"][0]["message"]
    # messages.append(bot_message)

    st.session_state["user_input"] = ""  # 入力欄を消去


# ユーザーインターフェイスの構築
st.title("AI概論サポートチャットへようこそ")
st.write("AI概論サポートチャットボットです。")
user_input = st.text_input("メッセージを入力してください。", key="user_input", on_change=communicate)
if  "messages" in st.session_state:
    messages = st.session_state["messages"]

    for message in reversed(messages[1:]):  # 直近のメッセージを上に
        speaker = "🙂"
        if message["role"]=="assistant":
            speaker="🤖"

        st.write(speaker + ": " + message["content"])
