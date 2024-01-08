import openai
import streamlit as st
from streamlit_chat import message
import speech_recognition as sr
import sys

sys.path.append('/')

from KnowledgeBase_PDF import openai_settings as settings

# Setting page title and header
st.set_page_config(page_title="CMA - GenAI-BOT", page_icon='./CGI_compressed_logo.png')
st.markdown("<h1 style='text-align: center;'>CMA - GenAI-BOT</h1>", unsafe_allow_html=True)

# Set org ID and API key
# openai.organization = os.getenv('OPENAI_ORGANIZATION')
openai.api_key = settings.API_KEY
openai.api_version = settings.API_VERSION
openai.api_base = settings.API_BASE
openai.api_type = settings.API_TYPE

# GUI
user_avatar = 'app/static/unisex_avatar.png'
bot_avatar = 'app/static/logo_cgi_color.png'

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# model = "WBU-GPT-4"
model = "WBU-GPT-35"


def take_voice_input():
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    return r.recognize_google(audio)


# generate a response
def generate_response(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})

    completion = openai.ChatCompletion.create(
        engine=model,
        messages=st.session_state['messages']
    )
    response = completion.choices[0].message.content
    st.session_state['messages'].append({"role": "assistant", "content": response})

    # print(st.session_state['messages'])
    total_tokens = completion.usage.total_tokens
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    return response, total_tokens, prompt_tokens, completion_tokens


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    st.session_state['messages'].append(
        {'role': 'assistant', 'content':
            """This is an Assistant of Canadian medical association Journal."""
         })

    st.session_state['messages'].append(
        {'role': 'system', 'content': """
        Here is a weblink provided in curly bracket that contain information about Canadian Medical Association Journal.; \
        url links: {https://www.cmaj.ca/content};

        You are a professional chatbot for Canadian medical association Journal. Answer questions based on the content of the link that I provided above. \
        While answering, please try to use numebr such as percentage (%), descriptive statistics, population etc. to make your answer looks mathmatically right. \
        Please use bullet points while answering and provide reference and url link of the subsection that you used to answer the question from the provided links. \
        At the beginning, start your conversation with greetings. 
        """}
    )

    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')
        microphone_button = st.form_submit_button(label='Voice_input')

    if submit_button and user_input:
        output, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
    if microphone_button:
        voice_registered = take_voice_input()
        print(voice_registered)
        output, total_tokens, prompt_tokens, completion_tokens = generate_response(voice_registered)
        st.session_state['past'].append(voice_registered)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', logo=f'{user_avatar}')
            message(st.session_state["generated"][i], key=str(i), logo=f'{bot_avatar}')
