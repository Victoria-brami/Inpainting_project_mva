#!/bin/bash

echo "Downloading the completion models..."

wget http://iizuka.cs.tsukuba.ac.jp/data/completionnet_places2.t7 -O completionnet_places2.t7
wget http://iizuka.cs.tsukuba.ac.jp/data/completionnet_places2_freeform.t7 -O completionnet_places2_freeform.t7
wget http://iizuka.cs.tsukuba.ac.jp/data/completionnet_celeba.t7 -O completionnet_celeba.t7

echo -e "Download finished!"

echo "Create directories and move checkpoints..."
mkdir checkpoints
mv completionnet_places2.t7 checkpoints/
mv completionnet_places2_freeform.t7 checkpoints/
mv completionnet_celeba.t7 checkpoints/

echo -e "All preparation done!"

