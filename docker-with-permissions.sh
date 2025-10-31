#!/bin/bash

# Helper script to run docker-compose with proper permissions
# This wraps docker-compose commands with the docker group

echo "🔧 Running docker-compose with docker group permissions..."

# Use sg to run the command with docker group
sg docker -c "docker-compose $@"

exit $?

