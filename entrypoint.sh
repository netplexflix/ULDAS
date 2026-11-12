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

    # Intel GPU (/dev/dri): collect the render nodes' group ids so a non-root
    # PUID can reach the GPU. gosu "uid:gid" keeps only the primary group, so
    # Docker's --group-add / group_add would otherwise be dropped.
    GPU_GIDS=""
    for node in /dev/dri/renderD* /dev/dri/card*; do
        [ -e "$node" ] || continue
        NODE_GID=$(stat -c '%g' "$node")
        case ",$PGID$GPU_GIDS," in *",$NODE_GID,"*) continue ;; esac
        GPU_GIDS="$GPU_GIDS,$NODE_GID"
    done

    # Create cache and log directories, set ownership
    mkdir -p /app/.cache /app/config/logs
    chown -R "$PUID:$PGID" /app/config /app/.cache

    if [ -n "$GPU_GIDS" ]; then
        echo "Running as $USER_NAME (PUID=$PUID, PGID=$PGID, GPU groups ${GPU_GIDS#,})"
        HOME=$(getent passwd "$PUID" | cut -d: -f6)
        export HOME
        exec setpriv --reuid="$PUID" --regid="$PGID" --groups="$PGID$GPU_GIDS" \
            python ULDAS.py "$@"
    fi
    echo "Running as $USER_NAME (PUID=$PUID, PGID=$PGID)"
    exec gosu "$PUID:$PGID" python ULDAS.py "$@"
else
    echo "Running as root"
    exec python ULDAS.py "$@"
fi