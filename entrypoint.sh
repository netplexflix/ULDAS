#!/bin/bash

PUID=${PUID:-0}
PGID=${PGID:-0}

# Check if config file exists, if not create it from example stored in /app
if [ ! -f /app/config/config.yml ]; then
    echo "Config file not found, creating from example..."
    mkdir -p /app/config
    mv /app/config.example.yml /app/config/config.yml
fi

# Create group and user if PUID/PGID are set to non-root
if [ "$PUID" != "0" ] && [ "$PGID" != "0" ]; then
    # Check if group with this GID exists, if not create it
    if ! getent group "$PGID" > /dev/null 2>&1; then
        groupadd -g "$PGID" uldas
        GROUP_NAME="uldas"
    else
        # Use existing group name for this GID
        GROUP_NAME=$(getent group "$PGID" | cut -d: -f1)
    fi

    # Check if user with this UID exists, if not create it
    if ! getent passwd "$PUID" > /dev/null 2>&1; then
        useradd -u "$PUID" -g "$PGID" -d /app -s /bin/bash -o uldas 2>/dev/null
        USER_NAME="uldas"
    else
        # Use existing user name for this UID
        USER_NAME=$(getent passwd "$PUID" | cut -d: -f1)
    fi

    # Create cache directory and set ownership
    mkdir -p /app/.cache
    chown -R "$PUID:$PGID" /app/config /app/.cache

    echo "Running as $USER_NAME (PUID=$PUID, PGID=$PGID)"
    exec gosu "$PUID:$PGID" python ULDAS.py "$@"
else
    echo "Running as root"
    exec python ULDAS.py "$@"
fi
