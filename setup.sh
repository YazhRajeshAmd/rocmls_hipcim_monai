#!/bin/bash

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e  # Exit on any error

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Paths
SOURCE_DIR="/scratch/users/rocm-ls/demo"
DEST_DIR="$(pwd)/"

print_status "Source: $SOURCE_DIR"
print_status "Destination: $DEST_DIR"

# Check source exists
if [ ! -d "$SOURCE_DIR" ]; then
    print_error "Source directory not found: $SOURCE_DIR"
    exit 1
fi



print_status "using cp..."
sudo cp -r "$SOURCE_DIR" "$DEST_DIR/"


print_success "Demo folder copied to $DEST_DIR"

# Show summary
file_count=$(find "$DEST_DIR" -type f | wc -l)
total_size=$(du -sh "$DEST_DIR" 2>/dev/null | cut -f1 || echo "unknown")
print_status "Copied files: $file_count, Size: $total_size"
