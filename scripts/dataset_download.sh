mkdir -p /dev/shm/teutonic/datasets_eval

seq -f "%06g" 1101 1101 | \
xargs -n 1 -P 8 -I {} \
wget -q --show-progress -c --tries=10 --timeout=30 \
  -O /dev/shm/teutonic/datasets_eval/shard_{}.npy \
  "https://s3.hippius.com/teutonic-sn3/dataset/v2/shards/shard_{}.npy"
