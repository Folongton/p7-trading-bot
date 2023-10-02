FROM tensorflow/tensorflow:2.14.0-gpu-jupyter

RUN apt-get update
RUN pip install --upgrade pip
# COPY requirements_NO_tf.txt requirements_NO_tf.txt
# RUN pip install -r /p7-trading-bot/.devcontainer/requirements_NO_tf.txt

# tensorflow's Jupyter default port
EXPOSE 8888 
# streamlit default port
EXPOSE 8501


# WORKDIR /p7-trading-bot-cont/src
# ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root","--no-browser"]
# ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# CMD ["sleep", "infinity"]