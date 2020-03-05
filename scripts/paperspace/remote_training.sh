
# Build the Docker image
docker build -t iosifmailo/curiosity .
docker push iosifmailo/curiosity

# Start Gradient job
gradient experiments run singlenode \
  --command 'cd /src && python app/sm_2d/train.py' \
  --experimentEnv "{\"RL_STORAGE\": \"/artifacts\"}" \
  --container iosifmailo/curiosity:latest \
  --machineType GPU+ \
  --name PyTorch \
  --ports 5000:5000 \
  --projectId prxdfyuy7

