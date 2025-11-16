# Dockerfile for Electron App
# This allows running Electron in a Docker container with X11 forwarding

FROM node:20-slim

# Install Electron dependencies and X11 libraries
RUN apt-get update && apt-get install -y \
    # X11 and display libraries
    libx11-dev \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libxkbcommon0 \
    # MESA software rendering drivers (for SwiftShader/llvmpipe)
    mesa-utils \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    # Utilities
    xvfb \
    x11-xserver-utils \
    # Emoji fonts for flag rendering
    fonts-noto-color-emoji \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    # Create dummy ALSA config to suppress warnings
    && mkdir -p /etc/alsa/conf.d \
    && echo 'pcm.!default { type null }' > /etc/alsa/conf.d/99-null.conf \
    && echo 'ctl.!default { type null }' >> /etc/alsa/conf.d/99-null.conf

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./
COPY electron/ ./electron/

# Install dependencies (as root)
RUN npm install

# Fix Electron sandbox permissions (required for running in Docker)
# The chrome-sandbox binary needs to be owned by root with SUID bit set
RUN if [ -f /app/node_modules/electron/dist/chrome-sandbox ]; then \
      chown root:root /app/node_modules/electron/dist/chrome-sandbox && \
      chmod 4755 /app/node_modules/electron/dist/chrome-sandbox; \
    fi

# Copy application files
COPY . .

# Set ownership to node user (node:20-slim comes with node user)
# But keep chrome-sandbox owned by root
RUN chown -R node:node /app && \
    if [ -f /app/node_modules/electron/dist/chrome-sandbox ]; then \
      chown root:root /app/node_modules/electron/dist/chrome-sandbox && \
      chmod 4755 /app/node_modules/electron/dist/chrome-sandbox; \
    fi

# Set display environment variable (will be overridden by docker-compose)
ENV DISPLAY=:0

# Expose port for VNC (optional, if using VNC instead of X11)
EXPOSE 5900

# Switch to non-root user (node user comes with node:20-slim image)
USER node

# Default command - can be overridden
# Always use --no-sandbox flag when running in Docker
# Disable GPU and Vulkan for containerized environments
CMD ["npx", "electron", ".", "--no-sandbox", "--disable-setuid-sandbox", "--disable-gpu", "--disable-vulkan"]

