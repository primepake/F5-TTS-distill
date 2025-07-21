#!/bin/bash
data_url=www.openslr.org/resources/60
data_dir=libritts


echo "Data Download"
for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
  ./download_and_untar.sh ${data_dir} ${data_url} ${part}
done
