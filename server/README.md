docker build -f common/base.Dockerfile -t vla-base:latest .

docker compose build

docker compose up -d

curl -X POST http://localhost:7001/step \
     -H "Content-Type: application/octet-stream" \
     --data-binary @frame.npy