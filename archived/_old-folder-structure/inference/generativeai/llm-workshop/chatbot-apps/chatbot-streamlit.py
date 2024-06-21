import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from typing import Any, Dict, List, Optional
import json
from io import StringIO, BytesIO
from random import randint
from PIL import Image
import boto3
import numpy as np
import pandas as pd
import json
import os
import base64
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

client = boto3.client('runtime.sagemaker')
aws_region = boto3.Session().region_name
source = []
st.set_page_config(page_title="Document Analysis", page_icon=":robot:")


Falcon_endpoint_name = os.getenv("nlp_ep_name", default="falcon-7b-instruct-2xl")
embedding_endpoint_name = os.getenv("embed_ep_name", default="huggingface-textembedding-all-MiniLM-L6-v2-2xlarge")



################# Prepare for RAG solution #######################
class SagemakerEndpointEmbeddingsJumpStart(SagemakerEndpointEmbeddings):
    def embed_documents(self, texts: List[str], chunk_size: int = 5) -> List[List[float]]:
        """Compute doc embeddings using a SageMaker Inference Endpoint.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size defines how many input texts will
                be grouped together as request. If None, will use the
                chunk size specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        results = []
        _chunk_size = len(texts) if chunk_size > len(texts) else chunk_size

        for i in range(0, len(texts), _chunk_size):
            response = self._embedding_func(texts[i : i + _chunk_size])
            print
            results.extend(response)
        return results


class ContentHandlerEmbed(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
        input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        embeddings = response_json["embedding"]
        return embeddings


content_handler_embed = ContentHandlerEmbed()

embeddings = SagemakerEndpointEmbeddingsJumpStart(
    endpoint_name=embedding_endpoint_name,
    region_name=aws_region,
    content_handler=content_handler_embed,
)

@st.cache_resource
def generate_index():
    loader = DirectoryLoader("./data/demo-video-sagemaker-doc/", glob="**/*.txt")
    documents = loader.load()
    docsearch = FAISS.from_documents(documents, embeddings)
    return docsearch

docsearch = generate_index()



################# sanity check for rag solution ########################

def create_sanity_prompt(
    instruction_start: str = None,
    instruction_end: str = None,
) -> PromptTemplate:
    """
    Create a prompt template for LLM sanity check

    Parameters
    ----------
    instruction_start : str, optional
        Instrcution in the beginning of the prompt, by default None
    instruction_end : str, optional
        Instrcution in the end of the prompt, by default None

    Returns
    -------
    PromptTemplate
        Prompt template in the LangChain format
    """

    # first instruction
    prompt_template = instruction_start + "\n"

    # add context
    prompt_template += "Context: {context}" + "\n"

    # add statement
    prompt_template += "Statement: {statement}" + "\n"

    # addinstruction
    prompt_template += "Question: " + instruction_end + "\n"

    # add answer placeholder
    prompt_template += "Answer:"

    # build the template
    llm_prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "statement"],
    )
    return llm_prompt



################# Prepare for chatbot with memory #######################

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"
    len_prompt = 0

    def transform_input(self, prompt: str, model_kwargs: Dict={}) -> bytes:
        self.len_prompt = len(prompt)
        input_str = json.dumps({"inputs": prompt, "parameters":{"max_new_tokens": st.session_state.max_token, "temperature":st.session_state.temperature, "seed":st.session_state.seed, "stop": ["Human:"], "num_beams":1, "return_full_text": False, "repetition_penalty": 1.9}})
        print(input_str)
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        response_json = output.read()
        res = json.loads(response_json)
        print(res)
        ans = res['generated_text']
        
        return ans[:ans.rfind("\nUser")].strip()


    
content_handler = ContentHandler()

llm = SagemakerEndpoint(
            endpoint_name=Falcon_endpoint_name,
            region_name=aws_region,
            content_handler=content_handler,
    )

@st.cache_resource
def load_chain(endpoint_name: str=Falcon_endpoint_name):

    memory = ConversationBufferMemory(return_messages=True)
    chain = ConversationChain(llm=llm, memory=memory)
    return chain

chatchain = load_chain()


# initialise session variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
    chatchain.memory.clear()
if 'rag' not in st.session_state:
    st.session_state['rag'] = False
if 'widget_key' not in st.session_state:
    st.session_state['widget_key'] = str(randint(1000, 100000000))
if 'max_token' not in st.session_state:
    st.session_state.max_token = 200
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.1
if 'seed' not in st.session_state:
    st.session_state.seed = 0
if 'extract_audio' not in st.session_state:
    st.session_state.extract_audio = False
if 'option' not in st.session_state:
    st.session_state.option = "NLP"
    
def clear_button_fn():
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['widget_key'] = str(randint(1000, 100000000))
    st.widget_key = str(randint(1000, 100000000))
    st.session_state.extract_audio = False
    chatchain.memory.clear()
    # chatchain = load_chain(endpoint_name=endpoint_names['NLP'])
    # chatchain.memory.clear()
    uploaded_file=None
    st.session_state.option = "NLP"

    
    
def on_file_upload():
    st.session_state.extract_audio = True
    st.session_state['generated'] = []
    st.session_state['past'] = []
    # st.session_state['widget_key'] = str(randint(1000, 100000000))
    chatchain.memory.clear()
    
def prompt_hints(prompt_list):
    sample_prompt = []
    for prompt in prompt_list:
        sample_prompt.append( f"- {str(prompt)} \n")
    return ' '.join(sample_prompt)
        
prompts = {
                'rag':[
                    "what is the recommended way to first customize a foundation model?",
                    "if prompt engineering is not able to handle specific task, what approach can you use to handle domain-specific tasks?",
                    "Does fine-tuning change the weights of the model?",
                    "Which country has the largest population in the world?",
                    "how can digital assets facilitate customers' engagement?"
                  ],
               'file_prompt': [
                   'what does this document talk about?',
                   'how much was the net income?',
                   'based on the financial results, how you do see the future growth of Amazon'
               ],
                'default': [
                    "\n Based on the Input Text, answer the Question. Input Text: Canceling my banking and direct investing account to move to your competitor. You've lost a long time customer. \n\n Question: what is the sentiment of the input text? \n\n Options: positive neutral negative. \n\n Answer:",
                    "write a kind message to the customer who provided the feedback"
                ]
              }
prompt_suggstion = prompt_hints(prompts['default'])

with st.sidebar:
    # Sidebar - the clear button is will flush the memory of the conversation
    st.sidebar.title("Conversation setup")
    clear_button = st.sidebar.button("Clear Conversation", key="clear", on_click=clear_button_fn)

    # upload file button
    uploaded_file = st.sidebar.file_uploader("Upload a text file", 
                                             key=st.session_state['widget_key'], 
                                             on_change=on_file_upload,
                                            )
    if uploaded_file:
        filename = uploaded_file.name
        print(filename)
        st.session_state.rag = False
        prompt_suggstion = prompt_hints(prompts['file_prompt'])
            
            
    rag = st.checkbox('Use knowledge base (answer question based on the retrieved relevant information from the video data source)', key="rag")

    if rag:
        prompt_suggstion = prompt_hints(prompts['rag'])
    st.sidebar.markdown(f'### Suggested prompts: \n\n {prompt_suggstion}')
            
    

    
left_column, _, right_column = st.columns([50, 2, 20])

with left_column:
    st.header("Building a multifunctional chatbot with Amazon SageMaker")
    # this is the container that displays the past conversation
    response_container = st.container()
    # this is the container with the input text box
    container = st.container()
    
    with container:
        # define the input text box
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("Input text:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')
        
        
        # when the submit button is pressed we send the user query to the chatchain object and save the chat history
        if submit_button and user_input:
            
            if rag:                    
                docs = docsearch.similarity_search_with_score(user_input)
                contexts = []
                for doc, score in docs:
                    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
                    if score <= 0.9:
                        contexts.append(doc)
                        source.append(doc.metadata['source'])
                print(f"\n INPUT CONTEXT:{contexts}")
                prompt_template = """Use the following pieces of Context to answer the Question at the end. If you don't know the answer, just say you don't know, don't try to make up an answer. Context:\n\n{context}\n\nQuestion: {question}\nHelpful Answer:"""

                PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                chain = load_qa_chain(llm=llm, prompt=PROMPT)
                output = chain({"input_documents": contexts, "question": user_input},
                               return_only_outputs=True)["output_text"]
                # create sanity check prompt
                sanity_prompt = create_sanity_prompt(
                    instruction_start="""The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Falcon, and a human user asking Questions. In the following interactions, Falcon will converse in natural language, and Falcon will answer the questions based only on the provided Context. Falcon will provide accurate, short and direct answers to the questions.""",
                    instruction_end="Is the above statement based directly on the provided context? Answer with yes or no.",
                )
                sanity_chain = load_qa_chain(llm=llm, prompt=sanity_prompt)
                sanity_check = sanity_chain({"input_documents": contexts, "statement": output},
                               return_only_outputs=True)['output_text']
                if not contexts or (sanity_check[0] == 'N'):
                    output = "Sorry. There is not enough information in the knowledge base for me to answer this question. Please try to ask another queston."
            else:
                output = chatchain(user_input)["response"]
            print(output)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
        # when a file is uploaded we also send the content to the chatchain object and ask for confirmation
        elif uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            content = "=== BEGIN FILE ===\n"
            content += stringio.read().strip()
            content += "\n=== END FILE ===\n\n Instruction: Please confirm that you have read that file by saying: 'Yes, I have read the file'"
            output = chatchain(content)["response"]
            st.session_state['past'].append("I have uploaded a file. Please confirm that you have read that file.")
            st.session_state['generated'].append(output)
            
        if source:
            df = pd.DataFrame(source, columns=['knowledge source'])
            st.data_editor(df)
            source = []    

        st.write(f"Currently using a {st.session_state.option} model")


    # this loop is responsible for displaying the chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
                

with right_column:

    max_tokens= st.slider(
        min_value=8,
        max_value=1024,
        step=1,
        # value=200,
        label="Number of tokens to generate",
        key="max_token"
    )
    temperature = st.slider(
        min_value=0.1,
        max_value=2.5,
        step=0.1,
        # value=0.4,
        label="Temperature",
        key="temperature"
    )
    seed = st.slider(
        min_value=0,
        max_value=1000,
        # value=0,
        step=1,
        label="Random seed to use for the generation",
        key="seed"
    )

    
