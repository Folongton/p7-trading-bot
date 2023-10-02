FROM tensorflow/tensorflow:2.14.0-gpu-jupyter

RUN apt-get update

ARG USERNAME=folongton
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME










# RUN pip install --upgrade pip
# COPY requirements_NO_tf.txt requirements_NO_tf.txt
# RUN pip install -r /p7-trading-bot/.devcontainer/requirements_NO_tf.txt

# tensorflow's Jupyter default port
# EXPOSE 8888 
# # streamlit default port
# EXPOSE 8501


# WORKDIR /p7-trading-bot-cont/src
# ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root","--no-browser"]
# ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# CMD ["sleep", "infinity"]