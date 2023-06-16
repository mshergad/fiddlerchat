import pandas as pd
import streamlit as st
import os
import openai
import tiktoken
from scipy import spatial
import ast


openai.api_key = os.environ.get('OPENAI_API_KEY')
embeddings_path = "./all_1_8.csv"
df = pd.read_csv(embeddings_path)
# convert embeddings from CSV str type back to list type
df['embedding'] = df['embedding'].apply(ast.literal_eval)

CHAT_HISTORY = 'chat_history'
ANSWER = 'answer'

if "query" not in st.session_state:
    st.session_state["query"] = ""

if "dummy" not in st.session_state:
    st.session_state["dummy"] = "blabla"
if CHAT_HISTORY not in st.session_state:
    st.session_state[CHAT_HISTORY] = []

if ANSWER not in st.session_state:
    st.session_state[ANSWER] = None

# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
):
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int,
    introduction='Use the below documentation from the company Fiddler to answer the subsequent question. Also, if possible, give me the reference URLs according to the following instructions. The way to create the URLs is: if you are discussing a client method or an API reference add "https://docs.fiddler.ai/reference/" before the "slug" value of the document. If it is Guide documentation add "https://docs.fiddler.ai/docs/" before before the "slug" value of the document. Only use the value following "slug:" to create the URLs and do not use page titles for slugs. If you are using quickstart notebooks, do not generate references. Note that if a user asks about uploading events, it means the same as publishing events. If the answer cannot be found in the documentation, write "I could not find an answer."'

):
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = string
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
    # query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
    temperature: int = 0,
    # chat_history=None,
    introduction='Use the below documentation from the company Fiddler to answer the subsequent question. Also, if possible, give me the reference URLs according to the following instructions. The way to create the URLs is: if you are discussing a client method or an API reference add "https://docs.fiddler.ai/reference/" before the "slug" value of the document. If it is Guide documentation add "https://docs.fiddler.ai/docs/" before before the "slug" value of the document. Only use the value following "slug:" to create the URLs and do not use page titles for slugs. If you are using quickstart notebooks, do not generate references. Note that if a user asks about uploading events, it means the same as publishing events. If the answer cannot be found in the documentation, write "I could not find an answer."'

):
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    query = st.session_state.query
    chat_history = st.session_state[CHAT_HISTORY]
    if chat_history is None:
        chat_history = []
    chat_history.append("User Query: "+query)
    # query = "\n".join(chat_history)
    message = query_message(query, df=df, model=model, token_budget=token_budget, introduction = introduction)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about Fiddler documentation."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    response_message = response["choices"][0]["message"]["content"]
    chat_history.append("Bot response: "+response_message)
    st.session_state[ANSWER], st.session_state[CHAT_HISTORY] = response_message, chat_history
    return response_message, chat_history


# def on_enter_pressed():
#     st.write("Enter key pressed! Entered text:", st.session_state.dummy)
# Streamlit app
def main():
    st.image('poweredby.png', width=250)
    st.title("Fiddler Chatbot")



    # st.text_input("Enter some text", key="dummy", on_change= on_enter_pressed, )#kwargs={'text':st.session_state.dummy}
    # st.session_state.dummy=text_val
    # Add a callback for the Enter key press
    # text_input_value.on_change(on_enter_pressed)

    # User input #user_input =
    st.text_input("You:", key="query", on_change=ask)

    # col1, _, col2 = st.columns([10, 1, 10])

    # with col1:
    #     st.write("test")
        # asking = st.button("Ask")

        # Generate response
        # if asking:
        #     if user_input.strip() != "":
        #         st.text("Bot:")
        #         st.session_state[ANSWER], st.session_state[CHAT_HISTORY] = ask(user_input, chat_history=st.session_state[CHAT_HISTORY])
        #     else:
        #         st.warning("Please enter a query.")
    # with col2:
    if st.button("Reset Chat History"):
        st.session_state[CHAT_HISTORY] = []

    if st.session_state[ANSWER] is not None:
        st.text("Bot:")
        st.write(st.session_state[ANSWER])

    # Display chat history
    st.header("Chat History")
    st.write("\n\n".join(st.session_state[CHAT_HISTORY]))

if __name__ == "__main__":
    main()


