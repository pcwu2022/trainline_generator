country="$1"

if [[ -z "$country" ]]; then
  echo "Usage: $0 <country>" >&2
  exit 1
fi

.venv/bin/python -m rail_gen run \
  --profile "$country" \
  --output "runs/${country}_production" \
  --iterations 200 \
  --batch-accept-limit 10 \
  --snapshot-every 1 \
  --gif-frame-ms 500 --overwrite