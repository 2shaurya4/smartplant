Team member names:
    - Shaurya Goel
    - Vardhan Jain

Instructions to compile the code:

    Step 1: Create a python virtual environment (Optional)
        - You can ignore this step, but this is how we did it.
        - On a Linux VM, open the folder with the source code in the terminal and 
        run "python3 -m venv venv" to make a virtual environment
        - Before doing anything else activate the environment by running 
        "source venv/bin/activate"
        - SSH into the RPi, create a folder where you'll later transfer the code
        files to. Open the folder in terminal, and run "python3 -m venv venv" to 
        create a VM

    Step 2: Install dependencies on both the VM and the RPi
        - If you created a virtual environment, then install all dependencies in
        venv, by activating it. To activate the venv, run "source venv/bin/activate"
        - To install dependencies on the VM, run "pip install paho-mqtt flask 
        flask-cors numpy scikit-learn pandas".  
        - SSH into the RPi and create a folder for the code files if you haven't already.
        Open the folder via the terminal.
        If you created a virtual environment on the RPi, then install all dependencies in
        venv, by activating it. To activate the venv, run "source venv/bin/activate"
        -To install dependencies on the RPi, run "pip3 install paho-mqtt 
        adafruit-circuitpython-dht". 

    Step 3: Install and configure mosquitto (The MQTT broker)
        - Mosquitto acts as the MQTT broker for this project
        - To install it run, "sudo apt-get install mosquitto mosquitto-clients"
        in the terminal, on the VM
        - By default mosquitto won't allow remote connections and will only start
        in "local mode". To fix this, create a mosquitto config file by running 
        "sudo nano /etc/mosquitto/conf.d/default.conf" and add 

        "listener 1883
        protocol mqtt
        allow_anonymous true

        listener 9001
        protocol websockets
        allow_anonymous true"
    
        to the config file by using your choice of text editor.
        - Start mosquitto by running, "sudo systemctl start mosquitto", 
        "sudo systemctl enable mosquitto"

    Step 4: Train the ML model
        - The ML classifier model used in this project was trained on hypothetical,
        ideal data. Hence, why we need to train it ourselves. 
        - To do so, run "python3 plant_ml_classifier.py" on the VM. If you created
        a venv at the start, make sure to activate it before you run any code.
        To activate the venv, run "source venv/bin/activate"
        - After running the code, you'll get a "plant_watering_model.pkl" file

    Step 5: Start the flask API server:
        - This piece of code uses MQTT to recieve data from the RPi.
        - If you created a venv at the start, make sure to activate it by running 
        "source venv/bin/activate"
        - To run the server file, run "python3 laptop_mqtt_server.py 
        --broker localhost --port 5000"
        - We're well aware that we didn't need to include arguments for
        the broker IP and the port number, but we did. This was a troubleshooting move
        which was left in the code. We didn't feel the need to remove it after troubleshooting.

    Step 6: Update the HTML dashboard and run it 
        - Open the HTML dashboard file in your choice of file editor.
        - Find the line that says, "const API_BASE_URL = 'http://192.168.56.101:5000'"
        - Find the IP address of your VM and swap it out.
        - It should look something like, "const API_BASE_URL = 'http://<YOUR_IP_ADDR>:5000'"
        - Open the dashboard in a web browser
        - We had the server running on the VM, but the dashboard was accessed via
        the host windows machine

    Step 7: Setup and run the code on the RPi
        - SSH into the RPi if you haven't already, then navigate to the folder
        where you have your code saved.
        - Run, "python3 rpi_plant_monitor.py --broker <YOUR_VM_IPADDR> --interval 5"
        - If you happen to have a DHT22 sensor then you can power it with 5V and connect the signal pin to pin 17
        of the RPi

    Step 8: Rejoice!
        - Everything should be up and running!

List of external libraries that were used:
    - paho-mqtt
    - flask
    - flask-cors
    - numpy
    - scikit-learn
    - pandas
    - adafruit-circuitpython-dht

Acknowledgement of AI/LLM tools used:
    - Claude Haiku 4.5 was used for the creation of the HTML dashboard, since
    neither of us have had any experience in writing HTML. Prompt used was

    "Create a responsive, real-time plant monitoring web dashboard in pure HTML5, 
    CSS3, and vanilla JavaScript (no external frameworks or jQuery). The dashboard 
    should poll a configurable Flask REST API base URL every 5 seconds to fetch live
    sensor data and machine learning predictions. Display four gauge 
    visualizations showing soil moisture (0-100%), temperature (10-35°C), 
    humidity (0-100%), and light level (0-100%), each with a numeric value, 
    progress bar, and min/max labels. Implement a large, color-coded watering 
    recommendation card that displays 'Don't Water' (green gradient), 'Water Soon' 
    (orange gradient), or 'Water Now' (red gradient) with an accompanying 
    confidence percentage. Include a header with system connection status 
    (green/orange/red pulsing dot indicator), last update timestamp, and an error 
    container for displaying API failures. Add a statistics section showing total 
    readings collected, watering recommendation counts, and historical aggregates. 
    Use a professional plant-themed color palette (purple/blue gradients for main 
    elements, green accents), modern sans-serif typography, and smooth CSS 
    transitions for gauge animations. 
    Include comprehensive error handling with try-catch blocks and user-friendly
    error messages, gracefully handle null/undefined data by displaying dashes, 
    and implement exponential backoff for failed API requests."

    - Claude Haiku 4.5 was used to troubleshoot and debug problems that we could not solve
    One such example of a troubleshoot prompt was:

    "I'm using mosquitto as the MQTT broker for a project, but it refuses to
    start up and this is the error I get. Can you help?

    1764403123: mosquitto version 2.0.18 starting
    1764403123: Using default config.
    1764403123: Starting in local only mode. Connections will only be possible from clients running on this machine.
    1764403123: Create a configuration file which defines a listener to allow remote access.
    1764403123: For more details see https://mosquitto.org/documentation/authentication-methods/
    1764403123: Opening ipv4 listen socket on port 1883.
    1764403123: Error: Address already in use
    1764403123: Opening ipv6 listen socket on port 1883.
    1764403123: Error: Address already in use"

    - Generally we'd give it chunks of problematic code to help us debug