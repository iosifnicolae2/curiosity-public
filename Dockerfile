FROM pytorch/pytorch

COPY ./__init__.py /src/__init__.py
COPY ./setup.py /src/setup.py
COPY ./app /src/app
COPY ./torch_ac /src/torch_ac
WORKDIR /src
RUN pip install -e .

CMD python app/sm_2d/train.py
