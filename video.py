import yt_dlp
from yt_dlp.postprocessor import FFmpegPostProcessor
FFmpegPostProcessor._ffmpeg_location.set(R'/Users/Fritz/Workspace/ffmpeg-6.1.1')








#//////////////////
from pydantic import BaseModel

from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core import SimpleDirectoryReader
from PIL import Image
import ffmpeg

from decouple import config
import secrets

import pandas as pd

import streamlit as st



GOOGLE_API_KEY = config("GOOGLE_API_KEY")
MODEL_NAME = "models/gemini-pro-vision"


class MeetingAttributes(BaseModel):
    """Data model of description of meeting"""
    number_people: int
    number_of_people_paying_attention: int
    who_is_nodding: str
    how_many_people_nod: str

prompt_template_str = """\
    Give me a summary of the meeting in the image\
    and return your respones with a json format\
"""


def structured_response_gemini(
    output_class: MeetingAttributes,
    image_documents: list,
    prompt_template_str: str,
    model_name: str = MODEL_NAME
):
    gemini_llm = GeminiMultiModal(
        api_key=GOOGLE_API_KEY,
        model_name=model_name
    )

    llm_program = MultiModalLLMCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_cls=output_class),
        video_documents=image_documents,
        prompt_template_str=prompt_template_str,
        multi_modal_llm=gemini_llm,
        verbose=True
    )

    response = llm_program()

    return response


def get_details_from_multimodal_gemini(uploaded_image):
    """get response"""
    for image_doc in uploaded_image:
        data_list = []
        structured_respose = structured_response_gemini(
            output_class=MeetingAttributes,
            image_documents=[image_doc],
            prompt_template_str=prompt_template_str,
            model_name=MODEL_NAME
        )
        print(structured_respose)
        for r in structured_respose:
            data_list.append(r)

        data_dict = dict(data_list)

        return data_dict


uploaded_file = st.file_uploader(
    "Choose An Image File",
    accept_multiple_files=False,
    type=["mp4"]
)


if uploaded_file is not None:
    st.toast("File uploaded successfully")
    byte_data = uploaded_file.read()
    st.write("Filename: ", uploaded_file.name)

    with st.spinner("Loading, please wait"):
        if uploaded_file.type == "video/mp4":
            file_type = "mp4"
        else:
            file_type = "mp4"

        # save file
        filename = f"{secrets.token_hex(8)}.{file_type}"

        with open(f"./HackDisability/videos/{filename}", "wb") as fp:
            fp.write(byte_data)

        file_path = f"./HackDisability/videos/{filename}"

        # load video
        image_documents = SimpleDirectoryReader(
            input_files=[file_path]
        ).load_data()

        response = get_details_from_multimodal_gemini(
            uploaded_image=image_documents
        )
        print(type(response))

        with st.sidebar:
            st.video(file_path) # caption=response.get()
            st.markdown(f"""
                    :green[Number of people]: :red[{response.get("number_people", "unknown")}]\n
                    :green[Number of people paying attention]: :violet[{response.get("number_of_people_paying_attention", "unknown")}]\n
                    :green[Focused Activities]: :orange[{response.get("who_is_nodding", "unknown")}]\n
                    :green[Distracted Activities]: :orange[{response.get("how_many_people_nod", "unknown")}]\n
                        """)
            
