FROM debian:trixie-slim AS download

ARG LLMMAN_VERSION=0.1.336
ARG LLAMA_CPP_VERSION=b10293
ARG TARGETARCH

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && case "$TARGETARCH" in \
         amd64) \
           target=x86_64-unknown-linux-gnu; \
           llama_asset=ubuntu-x64.tar.gz; \
           llama_sha256=dfca6141b016d1557297ba960861ed8bf43f9e18e405c0e8aaad35fbdf76ba9d \
           ;; \
         arm64) \
           target=aarch64-unknown-linux-gnu; \
           llama_asset=ubuntu-arm64.tar.gz; \
           llama_sha256=f996d1f336bb5f96e04bd7a4de9699d88c9ce09885349bdc92652aea2f1f30d3 \
           ;; \
         *) echo "unsupported architecture: $TARGETARCH" >&2; exit 1 ;; \
       esac \
    && base="https://github.com/llmmanorg/llmman/releases/download/v${LLMMAN_VERSION}" \
    && curl -fsSL --retry 16 --retry-all-errors \
         "$base/checksums.txt" -o /tmp/checksums.txt \
    && curl -fsSL --retry 16 --retry-all-errors \
         "$base/llmman-$target" -o "/tmp/llmman-$target" \
    && cd /tmp \
    && grep "  llmman-$target$" checksums.txt | sha256sum --check --strict \
    && mv "llmman-$target" /usr/local/bin/llmman \
    && chmod 0755 /usr/local/bin/llmman \
    && llama_archive="llama-${LLAMA_CPP_VERSION}-bin-${llama_asset}" \
    && llama_base="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP_VERSION}" \
    && curl -fsSL --retry 16 --retry-all-errors \
         "$llama_base/$llama_archive" -o "/tmp/${llama_archive}" \
    && echo "$llama_sha256  /tmp/$llama_archive" | sha256sum --check --strict \
    && mkdir -p /tmp/llama /opt/llama.cpp \
    && tar -xf "/tmp/${llama_archive}" -C /tmp/llama \
    && llama_server="$(find /tmp/llama -type f -name llama-server -print -quit)" \
    && test -n "$llama_server" \
    && cp -a "$(dirname "$llama_server")/." /opt/llama.cpp/

FROM debian:trixie-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 llmman \
    && mkdir -p /var/lib/llmman/store \
    && chown -R llmman:llmman /var/lib/llmman

COPY --from=download /usr/local/bin/llmman /usr/local/bin/llmman
COPY --from=download /opt/llama.cpp /opt/llama.cpp

USER llmman
ENV LLMMAN_HOST=0.0.0.0:17434 \
    LLMMAN_MODELS=/var/lib/llmman/store \
    LD_LIBRARY_PATH=/opt/llama.cpp \
    PATH=/opt/llama.cpp:$PATH
VOLUME ["/var/lib/llmman"]
EXPOSE 17434

ENTRYPOINT ["llmman"]
CMD ["serve"]
