# launch.py
import os
import subprocess
from pyngrok import ngrok
import webbrowser   
import time

# --- Your ngrok Authtoken ---
# IMPORTANT: Paste your authtoken here. You can get one from https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTHTOKEN = "30j96MzF00uj4BEXpMS4urTTa0T_4gyRc7CMDnNkp2hkx2ow3"

def main():
    """
    Starts the Streamlit app and creates a public ngrok tunnel to it.
    """
    if NGROK_AUTHTOKEN == "YOUR_NGROK_AUTHTOKEN_HERE":
        print("🔴 ERROR: Please paste your ngrok authtoken into the launch.py script.")
        return

    public_url_obj = None 
    
    print("[OK] Setting up ngrok authtoken...")
    ngrok.set_auth_token(NGROK_AUTHTOKEN)

    print("[START] Starting Streamlit app in the background...")
    # Runs the command: streamlit run app.py
    process = subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"])

    try:
        print("[INFO] Creating ngrok tunnel...")
        # Start an HTTP tunnel to the default Streamlit port 8501
        public_url_obj = ngrok.connect(8501, "http")
        
        # Extract the public URL string from the ngrok object
        public_url_str = public_url_obj.public_url
        
        print(f"\n>>> Your public URL is: {public_url_str}")
        print("    You can share this link with your reviewers.\n")
        
        # Open the URL in the default web browser
        time.sleep(2) 
        webbrowser.open(public_url_str)
        
        print("[INFO] Your Streamlit app is running. Keep this terminal open.")
        print("       Press Ctrl+C in this window to stop the server and close the tunnel.")
        
        # Wait for the Streamlit process to terminate
        process.wait()

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down server and tunnel.")
    finally:
        if public_url_obj:
            ngrok.disconnect(public_url_obj.public_url)
        process.terminate()
        print("[OK] Tunnel closed. Server stopped.")

if __name__ == "__main__":
    main()