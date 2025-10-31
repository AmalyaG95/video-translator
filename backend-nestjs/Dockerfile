FROM node:20-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build application
RUN npm run build

# Production image
FROM node:20-alpine

# Install FFmpeg for video streaming
RUN apk add --no-cache ffmpeg

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install production dependencies only
RUN npm ci --only=production

# Copy built application
COPY --from=builder /app/dist ./dist

# Copy proto files (needed at runtime for gRPC)
COPY --from=builder /app/src/ml-client/proto ./dist/ml-client/proto

# Create uploads directory
RUN mkdir -p /app/uploads /app/artifacts

# Expose port
EXPOSE 3001

# Start server
CMD ["node", "dist/main"]

