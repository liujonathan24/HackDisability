FROM python:3.9

ENV GOOGLE_API_KEY = ""
ADD main.py .

RUN pip install requests llama_index Pillow pydantic python-decouple streamlit llama_index.multi_modal_llms.gemini 
RUN pip install llama_index.core

CMD ["streamlit", "run", "main.py"]
