FROM python:3.12.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      curl \
      git \
      openssh-client \
      tini \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash coder \
    && mkdir -p /workspace /opt/cardioshift /home/coder/.ssh \
    && chown -R coder:coder /workspace /opt/cardioshift /home/coder/.ssh \
    && chmod 0700 /home/coder/.ssh

COPY --chown=coder:coder --chmod=0600 coder/known_hosts /home/coder/.ssh/known_hosts

COPY requirements.lock requirements-notebook.lock requirements-demo.lock /opt/cardioshift/

RUN python -m venv /opt/cardioshift/core \
    && /opt/cardioshift/core/bin/pip install --no-cache-dir \
      -r /opt/cardioshift/requirements.lock \
      -r /opt/cardioshift/requirements-notebook.lock \
    && python -m venv /opt/cardioshift/demo \
    && /opt/cardioshift/demo/bin/pip install --no-cache-dir \
      -r /opt/cardioshift/requirements-demo.lock

USER coder
WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sleep", "infinity"]
