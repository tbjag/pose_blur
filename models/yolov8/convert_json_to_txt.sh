#!/bin/bash
# filepath: /home/wenjun/Lab/GAN_project/tanush_pose_blur/models/yolov8/convert_json_to_txt.sh

# Base directory containing train and test subdirectories
BASE_DIR="/media/Data_2/person-search/dataset/Image/SSM/labels"
CATEGORY_ID=0

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed. Please install with 'sudo apt install jq'"
    exit 1
fi

# Process files in train directory
process_directory() {
    local dir="$1"
    
    echo "Processing files in $dir directory..."
    
    # Find all JSON files in the directory
    for json_file in "$dir"/*.json; do
        if [ -f "$json_file" ]; then
            base_name=$(basename "$json_file" .json)
            txt_file="$dir/$base_name.txt"
            
            echo "Converting $json_file to $txt_file"
            
            # Extract bounding boxes and convert to TXT format
            bboxes=$(jq -r '.[]' "$json_file" | jq -s '.')
            
            # Create empty output file
            > "$txt_file"
            
            # Process each bounding box
            num_boxes=$(jq 'length' <<< "$bboxes")
            for (( i=0; i<num_boxes; i++ )); do
                box=$(jq ".[$i]" <<< "$bboxes")
                x1=$(jq '.[0]' <<< "$box")
                y1=$(jq '.[1]' <<< "$box")
                x2=$(jq '.[2]' <<< "$box")
                y2=$(jq '.[3]' <<< "$box")
                
                # Write to txt file
                echo "$CATEGORY_ID $x1 $y1 $x2 $y2" >> "$txt_file"
            done
        fi
    done
}

# Process train and test directories
for subdir in "train" "test"; do
    if [ -d "$BASE_DIR/$subdir" ]; then
        process_directory "$BASE_DIR/$subdir"
    else
        echo "Warning: Directory $BASE_DIR/$subdir not found, skipping"
    fi
done

echo "Conversion complete!"