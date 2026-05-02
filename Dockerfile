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

# ──────────────────────────────────────────────────────────────
# Runtime stage — minimal image
# ──────────────────────────────────────────────────────────────
FROM --platform=linux/amd64 alpine:3.19

WORKDIR /app

COPY --from=builder /app/server .

# Copy reference data files into the image so they're always available.
# This avoids needing a shared volume and keeps the image self-contained.
COPY resources/ ./resources/

ENV PORT=8080
EXPOSE 8080

CMD ["./server"]
