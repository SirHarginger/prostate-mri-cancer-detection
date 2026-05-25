#!/usr/bin/env bash
set -euo pipefail

# TCIA downloads are often distributed through a manifest/NBIA workflow rather
# than one stable direct file URL. Put a cluster-local manifest URL or file path
# here when ready.

DEST_DIR="${1:-nnunet_autosegmentation/data/raw/prostatex}"
MANIFEST_OR_URL="${2:-}"

mkdir -p "${DEST_DIR}"

if [[ -z "${MANIFEST_OR_URL}" ]]; then
  echo "Usage: $0 <destination_dir> <manifest_or_url>"
  echo "Destination default: nnunet_autosegmentation/data/raw/prostatex"
  echo "Get the official PROSTATEx download/manifest from:"
  echo "https://www.cancerimagingarchive.net/collection/prostatex/"
  exit 2
fi

if [[ "${MANIFEST_OR_URL}" =~ ^https?:// ]]; then
  wget -P "${DEST_DIR}" "${MANIFEST_OR_URL}"
else
  cp "${MANIFEST_OR_URL}" "${DEST_DIR}/"
fi

echo "Downloaded/copied source into ${DEST_DIR}"
