FROM pytorch/pytorch

COPY . /src
WORKDIR /src

RUN pip install -e .

CMD python app/sm_2d/train.py
