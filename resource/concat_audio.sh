#!/bin/bash
# Script to concatenate audio files using ffmpeg

# Method 1: Using concat demuxer (fast, but requires same codec)
echo "Creating concat list file..."
cat > concat_list.txt << EOF
file './1.mp3'
file './2.mp3'
file './3.mp3'
file './4.mp3'
file './5.mp3'
file './6.mp3'
file './7.mp3'
file './8.mp3'
EOF

echo "Concatenating audio files..."
ffmpeg -f concat -safe 0 -i concat_list.txt -c copy output.mp3

if [ $? -eq 0 ]; then
    echo "✓ Successfully created output.mp3"
    rm concat_list.txt
else
    echo "✗ Failed with concat demuxer, trying alternative method..."
    
    # Method 2: Using filter_complex (slower, but more compatible)
    ffmpeg -i 1.mp3 -i 2.mp3 -i 3.mp3 -i 4.mp3 -i 5.mp3 -i 6.mp3 -i 7.mp3 -i 8.mp3 \
           -filter_complex "[0:0][1:0][2:0][3:0][4:0][5:0][6:0][7:0]concat=n=8:v=0:a=1[out]" \
           -map "[out]" output.mp3
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully created output.mp3 using filter_complex"
    else
        echo "✗ Both methods failed. Please check your input files."
        exit 1
    fi
fi

