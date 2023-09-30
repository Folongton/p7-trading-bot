FROM tensorflow/tensorflow:latest-gpu

COPY . /p7-trading-bot-cont/

WORKDIR /p7-trading-bot-cont
RUN pip install --upgrade pip
RUN pip install --upgrade -r requirements.txt

# tensorflow default port
EXPOSE 8888 
# streamlit default port
# EXPOSE 8501

# WORKDIR /p7-trading-bot-cont/src
ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root","--no-browser"]
# ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# WORKDIR /
# CMD ["sleep", "infinity"]