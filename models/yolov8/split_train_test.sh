#!/bin/bash

# Source directory
SRC_DIR="/media/Data_2/person-search/dataset/Image/SSM"

# Create train and test directories
mkdir -p "$SRC_DIR/images/train/"
mkdir -p "$SRC_DIR/labels/train/"
mkdir -p "$SRC_DIR/images/test"
mkdir -p "$SRC_DIR/labels/test"

# Get all image files
IMAGE_FILES=$(find "$SRC_DIR" -maxdepth 1 -name "*.jpg" -o -name "*.png" -o -name "*.jpeg")
JSON_FILES=$(find "$SRC_DIR" -maxdepth 1 -name "*.json")

# Count total files
TOTAL_IMAGES=$(echo "$IMAGE_FILES" | wc -w)
TRAIN_COUNT=$(( TOTAL_IMAGES * 70 / 100 ))

# Shuffle and split images
echo "Splitting $TOTAL_IMAGES images: $TRAIN_COUNT for training, $((TOTAL_IMAGES-TRAIN_COUNT)) for validation"

# Create a shuffled list of images
IMAGE_LIST=$(echo "$IMAGE_FILES" | tr ' ' '\n' | shuf)

# Copy files to train/valid directories
COUNT=0
for img in $IMAGE_LIST; do
  base_name=$(basename "$img")
  json_name="${base_name%.*}.json"
  
  if [ $COUNT -lt $TRAIN_COUNT ]; then
    # Training set
    cp "$img" "$SRC_DIR/images/train/"
    # Check if corresponding JSON exists and copy it
    if [ -f "$SRC_DIR/$json_name" ]; then
      cp "$SRC_DIR/$json_name" "$SRC_DIR/labels/train/"
    fi
  else
    # Validation set
    cp "$img" "$SRC_DIR/images/test/"
    # Check if corresponding JSON exists and copy it
    if [ -f "$SRC_DIR/$json_name" ]; then
      cp  "$SRC_DIR/$json_name" "$SRC_DIR/labels/test/"
    fi
  fi
  
  COUNT=$((COUNT+1))
done

echo "Split complete. Train: $TRAIN_COUNT images, Valid: $((TOTAL_IMAGES-TRAIN_COUNT)) images"