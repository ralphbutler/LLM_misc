
cleanup() {
    echo "Terminating the application..."
    kill $app_pid
    kj chainl
    echo "Terminated the application..."
}

trap cleanup SIGINT SIGTERM  # trap termination signals (e.g., SIGINT, SIGTERM)

# -w watch mode   -h headless (does NOT start browser)
# -w seems bad because it thinks writing a python file changes the app and reloads it
chainlit run app.py -h &
app_pid=$!

sleep 2  # give the app time to start

# open -a firefox http://localhost:8000  ### HANGS when I start recording
open -a "Google Chrome" http://localhost:8000

wait $app_pid

