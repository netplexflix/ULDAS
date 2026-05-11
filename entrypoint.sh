#!/bin/bash

PUID=${PUID:-0}
PGID=${PGID:-0}

# Log container start for debugging restarts
echo "=========================================="
echo "ULDAS Container Starting"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Check if config file exists, if not create it from example stored in /app
if [ ! -f /app/config/config.yml ]; then
    echo "Config file not found, creating from example..."
    mkdir -p /app/config
    cp /app/config.example.yml /app/config/config.yml
fi

# Create group and user if PUID/PGID are set to non-root
if [ "$PUID" != "0" ] && [ "$PGID" != "0" ]; then
    # Check if group with this GID exists, if not create it
    if ! getent group "$PGID" > /dev/null 2>&1; then
        groupadd -g "$PGID" uldas
        GROUP_NAME="uldas"
    else
        GROUP_NAME=$(getent group "$PGID" | cut -d: -f1)
    fi

    # Check if user with this UID exists, if not create it
    if ! getent passwd "$PUID" > /dev/null 2>&1; then
        useradd -u "$PUID" -g "$PGID" -d /app -s /bin/bash -o uldas 2>/dev/null
        USER_NAME="uldas"
    else
        USER_NAME=$(getent passwd "$PUID" | cut -d: -f1)
    fi

    # Create cache and log directories, set ownership
    mkdir -p /app/.cache /app/config/logs
    chown -R "$PUID:$PGID" /app/config /app/.cache

    echo "Running as $USER_NAME (PUID=$PUID, PGID=$PGID)"
    exec gosu "$PUID:$PGID" python ULDAS.py "$@"
else
    echo "Running as root"
    exec python ULDAS.py "$@"
fi