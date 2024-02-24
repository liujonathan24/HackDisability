FROM python:3.9

ENV GOOGLE_API_KEY = "AIzaSyAjzI_qNEW16dBb_we3Ptky4eU8bZYh3S0"
ADD main.py .

RUN pip install requests llama_index Pillow pydantic python-decouple streamlit llama_index.multi_modal_llms.gemini 
RUN pip install llama_index.core

CMD ["streamlit", "run", "main.py"]