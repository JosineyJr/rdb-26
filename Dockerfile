# ──────────────────────────────────────────────────────────────
# Build stage
# ──────────────────────────────────────────────────────────────
FROM --platform=linux/amd64 golang:1.26-alpine AS builder

WORKDIR /app

# Download dependencies first (cached layer).
# go.sum may be empty/absent for pure-stdlib modules — the glob handles both cases.
COPY go.mod go.sum* ./
RUN go mod download

# Copy source and build a statically linked binary
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
  go build -ldflags="-s -w" -o server .

# Reuse the checked-in IVF index for normal builds; rebuild only if the artifact
# is missing from the build context. The raw references are removed from the
# builder so they cannot be copied into the runtime image by accident.
RUN if [ ! -s resources/index.bin ]; then go run cmd/build_index/main.go; fi && rm -f resources/references.json.gz

# ──────────────────────────────────────────────────────────────
# Runtime stage — minimal image
# ──────────────────────────────────────────────────────────────
FROM --platform=linux/amd64 alpine:3.19

WORKDIR /app

COPY --from=builder /app/server .

# Copy reference data files into the image so they're always available.
# This avoids needing a shared volume and keeps the image self-contained.
COPY --from=builder /app/resources/index.bin ./resources/index.bin
COPY --from=builder /app/resources/mcc_risk.json ./resources/mcc_risk.json
COPY --from=builder /app/resources/normalization.json ./resources/normalization.json

ENV PORT=8080
EXPOSE 8080

CMD ["./server"]
