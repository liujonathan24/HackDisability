from pydantic import BaseModel

from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core import SimpleDirectoryReader
from PIL import Image

from decouple import config
import secrets

import streamlit as st


GOOGLE_API_KEY = "AIzaSyAjzI_qNEW16dBb_we3Ptky4eU8bZYh3S0"
MODEL_NAME = "models/gemini-pro-vision"


class MeetingAttributes(BaseModel):
    """Data model of description of meeting"""
    number_people: int
    number_of_people_paying_attention: int
    what_focused_people_are_doing: str
    what_are_non_focused_people_doing: str
    distractions: str
    Describe_the_participants_in_the_meeting: str
    number_of_people_using_phone_or_not_in_the_photo_frame_or_looking_to_the_side_or_have_unusual_camera_angle_or_have_off_camera_activity_or_inattentive_body_language: int
    from_the_zoom_meeting_name_labels_what_are_the_names_of_the_participants_using_phones_or_not_in_the_photo_frame_or_looking_to_the_side_or_have_unusual_camera_angle_or_have_off_camera_activity_or_inattentive_body_language: str
    names: str
    name_of_top_left_participant_in_the_photo: str
    number_of_people_with_camera_off: int
    the_current_speaker_is_highlighted_by_a_green_yellow_box_Who_is_the_current_speaker: str

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
        image_documents=image_documents,
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
    type=["png", "jpg"]
)


if uploaded_file is not None:
    st.toast("File uploaded successfully")
    byte_data = uploaded_file.read()
    st.write("Filename: ", uploaded_file.name)

    with st.spinner("Loading, please wait"):
        if uploaded_file.type == "image/jpeg":
            file_type = "jpg"
        else:
            file_type = "png"

        # save file
        filename = f"{secrets.token_hex(8)}.{file_type}"

        with open(f"./hackdisability/images/{filename}", "wb") as fp:
            fp.write(byte_data)

        file_path = f"./hackdisability/images/{filename}"


        # load images
        image_documents = SimpleDirectoryReader(
            input_files=[file_path]
        ).load_data()

        response = get_details_from_multimodal_gemini(
            uploaded_image=image_documents
        )
        print(type(response))

        with st.sidebar:
            st.image(image=file_path) # caption=response.get()
            st.markdown(f"""
                    :green[Number of people]: :red[{response.get("number_people", "unknown")}]\n
                    :green[Number of people paying attention]: :violet[{response.get("number_of_people_paying_attention", "unknown")}]\n
                    :green[Focused Activities]: :orange[{response.get("what_focused_people_are_doing", "unknown")}]\n
                    :green[Distracted Activities]: :orange[{response.get("what_are_non_focused_people_doing", "unknown")}]\n
                    :green[Distractions]: :orange[{response.get("distractions", "unknown")}]\n
                    :green[num people]: :orange[{response.get("Describe_the_participants_in_the_meeting", "unknown")}]\n
                    :green[distracted people]: :orange[{response.get("number_of_people_using_phone_or_not_in_the_photo_frame_or_looking_to_the_side_or_have_unusual_camera_angle_or_have_off_camera_activity_or_inattentive_body_language", "unknown")}]\n
                    :green[not paying attention names]: :orange[{response.get("from_the_zoom_meeting_name_labels_what_are_the_names_of_the_participants_using_phones_or_not_in_the_photo_frame_or_looking_to_the_side_or_have_unusual_camera_angle_or_have_off_camera_activity_or_inattentive_body_language", "unknown")}]\n
                    :green[names]: :orange[{response.get("names", "unknown")}]\n
                    :green[top left names]: :orange[{response.get("name_of_top_left_participant_in_the_photo", "unknown")}]\n
                    :green[number of people with camera off]: :orange[{response.get("number_of_people_with_camera_off", "unknown")}]\n
                    :green[current speaker]: :orange[{response.get("the_current_speaker_is_highlighted_by_a_green_yellow_box_Who_is_the_current_speaker", "unknown")}]\n
                    
                    

                        """)
            
