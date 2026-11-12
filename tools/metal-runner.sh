#!/bin/sh
# Runs this repo's ignored GPU tests on request, so a remote session that can edit files in
# this folder — but cannot run anything on the Mac — can execute them and read the result.
#
#   sh tools/metal-runner.sh          # leave it running in a terminal; Ctrl-C stops it
#
# It watches .runner/request for one line: the name of a test (letters, digits, _ and :
# only). It then runs
#
#   cargo test --release --features metal <name> -- --ignored --nocapture
#
# with the output in .runner/log, and writes the exit status to .runner/done. Nothing else
# can be requested: no other command, no arguments.
cd "$(dirname "$0")/.." || exit 1
mkdir -p .runner
echo "watching .runner/request (Ctrl-C to stop)"
while :; do
    if [ -f .runner/request ]; then
        name=$(head -n 1 .runner/request | tr -d '\r\n')
        rm -f .runner/request .runner/done
        case "$name" in
            "" | *[!A-Za-z0-9_:]*)
                echo "refused: not a test name" | tee .runner/log
                echo "refused" > .runner/done
                ;;
            *)
                echo "$(date '+%H:%M:%S') running $name"
                cargo test --release --features metal "$name" -- --ignored --nocapture \
                    > .runner/log 2>&1
                status=$?
                echo "$status" > .runner/done
                echo "$(date '+%H:%M:%S') $name finished with status $status"
                ;;
        esac
    fi
    sleep 2
done
