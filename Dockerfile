FROM tensorflow/tensorflow:2.3.0-gpu

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# COPY app.py ${APP_HOME}/app.py
# COPY images ${APP_HOME}/images
# COPY temp1 ${APP_HOME}/temp1

COPY . .

# ENTRYPOINT [ "streamlit", "run"]
# CMD ["app.py"]
CMD streamlit run app.py